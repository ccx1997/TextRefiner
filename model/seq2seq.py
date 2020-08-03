import torch
from torch import nn
import torch.nn.functional as F
import math
import random
from model.ResNet import ResNet
from model.STN import STN
from model.selfattention import AttentionNet


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):    # x.size [batch, t, feature]
        N, seq_len, _ = x.size()
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))    # [batch, seq_len, hidden_size*num_directions]
        out = out.contiguous()
        out = out.view(-1, out.size(2))    # reshape to [batch*seq_length, hidden_size*2]
        out = self.fc(out).view(N, seq_len, -1)
        return out    # [b, t, c]


class Encoder(nn.Module):
    def __init__(self, lstm_input_size=512, hidden_size=256, num_layers=1, bidirectional=True, use_stn=False):
        super(Encoder, self).__init__()
        if use_stn:
            print('Creating model with STN')
            # self.features = nn.Sequential(STN(output_img_size=[32, 100], num_control_points=20, margins=[0.1, 0.1]), 
            #                               ResNet())
            self.stn = STN(output_img_size=[32, 100], num_control_points=20, margins=[0.1, 0.1])
        else:
            print('Creating model without STN')
        self.use_stn = use_stn
        self.features = ResNet()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bias=True, 
            batch_first=True, 
            bidirectional=bidirectional
        )

    def forward(self, x):
        recitified = None
        if self.use_stn:
            x = self.stn(x)
            recitified = x
        x = self.features(x)    # [B, C, H, W], H=1
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)    # [batch, t, channels]
        x, state = self.lstm(x)
        return x, recitified


class BahdanauAttentionMechanism(nn.Module):
    def __init__(self, query_dim, values_dim, attention_dim):
        super(BahdanauAttentionMechanism, self).__init__()
        self.fc_query = nn.Linear(query_dim, attention_dim)
        self.fc_values = nn.Linear(values_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, values):
        """
        Args:
            query (s_i): [batch_size, query_dim]
            values: [batch_size, T, values_dim]
        Returns:
            context: (c_i), [batch_size, values_dim]
            attention: (a_i), [batch_size, T]
        """
        keys = self.fc_values(values)
        query = self.fc_query(query)
        query = query.unsqueeze(1).expand_as(keys)
        e_i = self.v(self.tanh(query + keys)).squeeze(-1)
        a_i = self.softmax(e_i)
        c_i = torch.bmm(a_i.unsqueeze(1), values)
        c_i = c_i.squeeze(1)
        return c_i, a_i


class AttentionDecoder(nn.Module):

    def __init__(self, hidden_dim, attention_dim, y_dim, encoder_output_dim, f_lookup_ts=None):
        super(AttentionDecoder, self).__init__()
        embedding_dim = 200
        self.lstm_cell = nn.LSTMCell(embedding_dim+encoder_output_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, y_dim)
        self.attention_mechanism = BahdanauAttentionMechanism(
                query_dim=hidden_dim, values_dim=encoder_output_dim, attention_dim=attention_dim)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.y_dim = y_dim
        self.encoder_output_dim = encoder_output_dim

        lookup_ts = torch.load(f_lookup_ts)
        self.lookup_ts = torch.cat(
            (torch.zeros(1, embedding_dim), torch.ones(1, embedding_dim), lookup_ts), dim=0).cuda()
        # self.embedding = nn.Embedding(y_dim, embedding_dim)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim).cuda(), 
                torch.zeros(batch_size, self.hidden_dim).cuda())

    def forward(self, pre_y_token, pre_state_hc, encoder_output, need_refine=False):
        c, a = self.attention_mechanism(pre_state_hc[0], encoder_output)
        pre_y = self.lookup_ts[pre_y_token]
        # pre_y = self.embedding(pre_y_token)
        new_state_h, new_state_c = self.lstm_cell(
                torch.cat((pre_y, c), -1), pre_state_hc)
        out = self.fc_out(new_state_h)
        if need_refine:
            out = (out, c)
        return out, (new_state_h, new_state_c), a


class RefineText(nn.Module):
    """
    Process 1-d text recognition. Could be placed after any ocr model.
    Input: character-level features, probabilities and vanilla predictions got by Decoder
    """
    def __init__(self, num_classes, feat_dim, embedding_dim=200, f_lookup_ts=None, p_lower=0.95):
        super(RefineText, self).__init__()
        h_dim = feat_dim
        self.vis_emb = nn.Linear(feat_dim, h_dim)
        self.sequence_model = AttentionNet(input_size=embedding_dim, hidden_size=h_dim,
                                           depth=3, head=5, k=8, pretrain="param/attention2.pkl")
        self.sem_emb = nn.Linear(5 * h_dim, h_dim)
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes)
        )
        self.SoS = torch.zeros(1, embedding_dim).cuda()
        self.EoS = torch.ones(1, embedding_dim).cuda()
        self.null = self.SoS
        lookup_ts = torch.load(f_lookup_ts)
        self.lookup_ts = torch.cat((self.SoS, self.EoS, lookup_ts.cuda()), dim=0)
        self.num_classes = num_classes
        self.p_lower = p_lower

    def random_drop(self, embs, idx, drop_proportion=0.3, drop_rate=0.3):
        """only used in training stage
        :param embs: [b, T, 200]
        :param idx: an integer
        :param drop_proportion: max proportion of number to be dropped
        :param drop_rate: the likelihood of dropping
        """
        embs_tmp = embs + 0  # Avoid in-place value changes
        if random.random() < drop_rate:
            T = embs.size(1)
            index = list(range(T))
            random.shuffle(index)
            num_drop = random.randint(1, math.ceil(T * drop_proportion))
            embs_tmp[:, index[:num_drop]] = self.null
        embs_tmp[:, idx] = self.null
        return embs_tmp

    def target_sort(self, probs):
        """
        To find where the characters needing to be refined is and sort them by expectation confidence.
        :param probs: [T]
        :return:
        """
        tgt_idx = torch.nonzero(probs < self.p_lower)
        if tgt_idx.size(0) == 0:
            return None
        tgt_idx = tgt_idx.squeeze(1)
        if tgt_idx.size(0) == 1:
            return tgt_idx
        expectation_p = F.conv1d(probs.view(1, 1, -1), torch.tensor([[[0.2, 0.3, 0, 0.3, 0.2]]]).cuda(), padding=2)
        expectation_p = expectation_p[0, 0, tgt_idx]
        _, id_id = expectation_p.sort(descending=True)
        tgt_idx = tgt_idx[id_id]
        return tgt_idx

    def p_drop(self, embs, p, threshold=0.95):
        """
        drop some embeddings where p < threshold
        :param embs: [T, 200]
        :param p: [T]
        :return:
        """
        assert embs.size(0) == p.size(0)
        p = p.unsqueeze(1)
        p = (p > threshold)
        p = p.to(torch.float32)
        return embs * p

    def one_pass(self, bridge, embs, id_insert, p_vis):
        """
        Refine a specific character using its context and itself.
        :param bridge: a tensor to represent visual information, [b, T, 512]
        :param embs: embeddings where some are missing, [b, T, 200]
        :param id_insert: the location (id) to refine
        :param p_vis: confidence of visual prediction, [b,]
        :return: [b, nc]
        """
        fi = bridge[:, id_insert, :]  # [b, 512]
        # randomly dropout some fi to make semantic learnable
        # if self.training:
        #     likelihood = random.random()
        #     if likelihood < 0.1:
        #         fi = fi / 20
        id_insert = torch.ones(embs.size(0)).long().cuda() * id_insert
        context = self.sequence_model(embs, id_insert)  # [b, 512]
        context = self.sem_emb(context)
        score_fusion = self.classifier(fi*0.5+context*0.5)  # [b, nc]
        if self.training:
            score_vis = self.classifier(fi)
            score_sem = self.classifier(context)
            return score_vis, score_sem, score_fusion
        else:
            return score_fusion

    def forward(self, cf, cp, y0):
        """
        :param cf: character-level features, [b, T, c], T is the number of characters
        :param cp: character-level probabilities, [b, T]
        :param y0: character-level gt/predictions (id of character), [b, T]
        :return: scores with size [b, nc, T] for training; [b, T] for testing
        """
        if y0[0, -1] == 1:
            y0 = y0[:, :-1]  # do not include the EoS token
            cf = cf[:, :-1, :]
            cp = cp[:, :-1]
        assert cf.shape[:2] == cp.shape, "character-level features and probs must have the same batch and length"
        assert cp.shape == y0.shape, "character-level probs and gts must have the same batch and length"
        bridge = self.vis_emb(cf)   # [b, T, 512]
        b, T, c = cf.size()
        if self.training:
            order = list(range(T))  # [1, 2, ..., T-1]
            random.shuffle(order)
            scores = torch.zeros(b, self.num_classes, T).cuda()  # serve for loss
            scores_vis = torch.zeros(b, self.num_classes, T).cuda()
            scores_sem = torch.zeros(b, self.num_classes, T).cuda()
            embedding = self.lookup_ts[y0]  # [b, T, 200]
            for idx in iter(order):
                embedding = self.random_drop(embedding, idx)
                scorei = self.one_pass(bridge, embedding, idx, cp[:, idx])
                scores_vis[:, :, idx] = scorei[0]
                scores_sem[:, :, idx] = scorei[1]
                scores[:, :, idx] = scorei[2]
            return scores_vis, scores_sem, scores
        else:
            y0 = y0 + 0
            for i_sample, pi in enumerate(cp):
                tgt_idx = self.target_sort(pi)
                if tgt_idx is None:
                    break
                tgt_idx = tgt_idx.tolist()
                for idx in iter(tgt_idx):
                    embedding = self.lookup_ts[y0[i_sample]]  # [T, 200]
                    embedding = self.p_drop(embedding, cp[i_sample])
                    scorei = self.one_pass(bridge[i_sample].unsqueeze(0),
                                           embedding.unsqueeze(0), idx, pi[idx].unsqueeze(0))
                    pii = F.softmax(scorei, dim=1)  # [b, nc]
                    pii, yii = torch.max(pii, 1)
                    if pii[0] > cp[i_sample, idx]:
                        y0[i_sample, idx] = yii[0]
                        cp[i_sample, idx] = pii[0]
            return y0


if __name__ == '__main__':
    net = RefineText(num_classes=100, feat_dim=512, f_lookup_ts='/home/dataset/TR/synth_cn/lookup.pt')
    net.cuda()
    net.train()
    cf = torch.rand(2, 8, 512).cuda()
    cp = torch.rand(2, 8).cuda()
    y0 = torch.tensor([[4, 33, 54, 76, 23, 78, 87, 98], [4, 33, 54, 76, 23, 78, 87, 98]]).cuda()
    pred = net(cf, cp, y0)
    print(pred[0].shape)
