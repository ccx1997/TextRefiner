import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, depth=1):
        super(SelfAttention, self).__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.linear_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.linear_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.depth = depth

    def forward(self, x):
        """
        self attention.
        :param x: [b, T, input_size]
        :return: hs: [b, T, hidden_size]
        """
        hs = self.fc0(x)  # [b, T, hidden_size]
        for _ in range(self.depth):
            q = self.linear_q(hs)
            k = self.linear_k(hs)
            v = self.linear_v(hs)
            a = q.matmul(k.permute(0, 2, 1))  # [b, T, T]
            a = F.softmax(a, dim=2)
            hs = self.fc1(a.matmul(v))  # [b, T, hidden_size]
        return hs


class SelfAttentionRPR(SelfAttention):
    def __init__(self, input_size, hidden_size, k, pos_size=None, depth=1):
        """
        :param k: distant we would consider
        :param pos_size: size of embedding for every relative position
        :param depth: number of iterations
        """
        super(SelfAttentionRPR, self).__init__(input_size, hidden_size, depth)
        if pos_size is None:
            pos_size = hidden_size
        self.pos_embedding_key = nn.Parameter(torch.randn(k * 2 + 1, pos_size))
        self.pos_embedding_value = nn.Parameter(torch.randn(k * 2 + 1, pos_size))
        nn.init.kaiming_normal_(self.pos_embedding_key)
        nn.init.kaiming_normal_(self.pos_embedding_value)
        self.k = k

    def relative_pos_emb(self, T, pe):
        # Every row denotes a position
        base = torch.arange(self.k, self.k + T).repeat(T, 1)
        minus_d = torch.arange(T).unsqueeze(1)
        relative_mat_id = torch.clamp(base - minus_d, min=0, max=2 * self.k).to(pe.device)
        return pe[relative_mat_id.view(-1)].view(T, T, -1)

    def forward(self, x):
        """
        self attention.
        :param x: [b, T, input_size]
        :return: hs: [b, T, hidden_size]
        """
        T = x.size(1)
        rpr_key = self.relative_pos_emb(T, self.pos_embedding_key)
        rpr_value = self.relative_pos_emb(T, self.pos_embedding_value)
        hs = self.fc0(x)  # [b, T, hidden_size]
        for _ in range(self.depth):
            q = self.linear_q(hs)
            k = self.linear_k(hs)
            v = self.linear_v(hs)
            # query-key
            a = q.matmul(k.permute(0, 2, 1))  # [b, T, T]
            a_pos = rpr_key.matmul(q.unsqueeze(3)).squeeze(3)  # [b, T, T]
            a = a + a_pos
            a = F.softmax(a, dim=2)  # [b, T, T]
            # attention-value
            c = a.matmul(v)
            c_pos = a.unsqueeze(2).matmul(rpr_value).squeeze(2)  # [b, T, hidden_size]
            c = c + c_pos
            hs = self.fc1(c)  # [b, T, hidden_size]
        return hs


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, depth, head, k=None):
        """
        :param depth: number of iterations
        :param head: number of heads
        :param k: distant we would consider in rpr
        """
        super(MultiHeadAttention, self).__init__()
        if k is not None:
            self.attention = nn.ModuleList([SelfAttentionRPR(
                input_size, hidden_size, k, depth=depth) for _ in range(head)])
        else:
            self.attention = nn.ModuleList([SelfAttention(input_size, hidden_size, depth) for _ in range(head)])

    def forward(self, x):
        ys = []
        for m in self.attention:
            ys.append(m(x))
        return torch.cat(ys, dim=2)


class AttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, depth, head, k=None, pretrain=''):
        super(AttentionNet, self).__init__()
        self.mha = MultiHeadAttention(input_size, hidden_size, depth, head, k)
        self.use_rpr = True if k is not None else False
        if pretrain:
            self.load_state_dict(torch.load(pretrain), strict=False)
            print("Loaded pretrained parameters in Attention Net from " + pretrain)

    def forward(self, embedding, idx):
        """
        self attention.
        :param embedding: [b, T, d_emb]
        :param idx: [b,]
        """
        b, T, d_emb = embedding.size()
        device = embedding.device
        if not self.use_rpr:
            pos_emb = torch.arange(T).float().repeat(b, 1).unsqueeze(2) / T  # [b, T, 1]
            embedding = torch.cat((embedding, pos_emb.to(device)), dim=2)  # [b, T, d_emb + 1]
        hs = self.mha(embedding)
        hsi = hs[torch.arange(b).to(device), idx]  # [b, hidden_size*k]
        return hsi


class AttentionNet2(nn.Module):
    def __init__(self, input_size, hidden_size, depth, head, num_classes, f_lookup_ts=None, k=None):
        super(AttentionNet2, self).__init__()
        if f_lookup_ts is not None:
            lookup_ts = torch.load(f_lookup_ts)
            lookup_ts = torch.cat((torch.zeros(1, lookup_ts.size(1)), lookup_ts), dim=0)
            self.register_buffer("lookup_ts", lookup_ts)
        else:
            self.register_buffer("lookup_ts", torch.zeros(num_classes + 1, 200))
            print("Randomly initializing the lookup table. It is to be fed with pretrained model parameters.")
        self.mha = MultiHeadAttention(input_size, hidden_size, depth, head, k)
        self.cls = nn.Sequential(
            nn.Linear(hidden_size * head, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
        self.use_rpr = True if k is not None else False

    def idx2embedding(self, indexes):
        return self.lookup_ts[indexes]

    def forward(self, x, idx):
        """
        self attention.
        :param x: [b, T]
        :param idx: [b,]
        :return:
        """
        b, T = x.size()
        device = x.device
        embedding = self.idx2embedding(x)  # [b, T, d_emb]
        if not self.use_rpr:
            pos_emb = torch.arange(T).float().repeat(b, 1).unsqueeze(2) / T  # [b, T, 1]
            embedding = torch.cat((embedding, pos_emb.to(device)), dim=2)  # [b, T, d_emb + 1]
        hs = self.mha(embedding)
        y = self.cls(hs[torch.arange(b).to(device), idx])  # [b, num_classes]
        return y
