import os
import argparse
import torch
import torch.nn.functional as F
import cv2
from model.seq2seq import Encoder, AttentionDecoder
from model.selfattention import AttentionNet2
from torchvision import transforms
from data.load_data import get_alphabet, get_dataloader
from model_utils import Ids2Str, batch_test


def decorate_model(model, is_training, device):
    model = model.to(device)
    if is_training:
        model.train()
    else:
        model.eval()


def load_vanilla(num_classes, device, args):
    encoder = Encoder(use_stn=args.use_stn).to(device)
    decoder = AttentionDecoder(hidden_dim=256, attention_dim=256, y_dim=num_classes, encoder_output_dim=512,
                               f_lookup_ts="/home/dataset/TR/synth_cn/lookup.pt").to(device)  # y_dim for classes_num
    lm = AttentionNet2(input_size=200, hidden_size=512, depth=3, head=5, num_classes=num_classes-2, k=8)
    decorate_model(encoder, is_training=False, device=device)
    decorate_model(decoder, is_training=False, device=device)
    decorate_model(lm, is_training=False, device=device)
    checkpoint = torch.load(args.ocr)
    encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
    decoder.load_state_dict(checkpoint["state_dict"]["decoder"])
    lm.load_state_dict(torch.load(args.nlp))
    return encoder, decoder, lm


def get_loader(is_training, alphabet, bs):
    lmdb_dir = "/home/dataset/TR/synth_cn/train_lmdb/" if is_training else "/home/dataset/TR/synth_cn/test_lmdb/"
    input_size = [64, 256] if args.use_stn else [32, 128]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trsf = transforms.Compose([transforms.ToTensor(), normalize])
    dataloader = get_dataloader(lmdb_dir, alphabet, input_size, trsf, batch_size=bs, is_train=is_training)
    return dataloader


class RefineText2(object):
    """
    Process 1-d text recognition. Could be placed after any ocr model.
    Input: character-level features, probabilities and vanilla predictions got by Decoder
    """
    def __init__(self, lm, p_lower=0.9):
        self.lm = lm
        self.p_lower = p_lower

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

    def p_drop(self, y, p, idx, threshold=0.9):
        """
        drop some embeddings where p < threshold
        :param y: [T]
        :param p: [T]
        :return:
        """
        assert y.size(0) == p.size(0)
        switch = (p > threshold)
        switch = switch.long()
        y = y * switch
        y[idx] = 0
        return y

    def choose(self, scores, topki, topkp):
        scores_candidate = scores[0, topki]  # semantic, [k]
        probs = F.softmax(scores_candidate, dim=0)
        probs = 0.5 * (torch.exp(probs) - 1)
        topkp = topkp ** 2
        final = topkp * probs
        final = final / final.sum()
        return final

    def __call__(self, cy, cp, topki, topkp):
        """
        :param cy: character-level predictions, [b, T], T is the number of characters
        :param cp: character-level probabilities, [b, T]
        :param topki: character-level top-k predictions (id of character), [b, T, k]
        :param topkp: character-level top-k probabilities, [b, T, k]
        :return: scores with size [b, k, T] and results with size [b, T]
        """
        if cy[0, -1] == 1:
            cy = cy[:, :-1]  # do not include the EoS token
            cp = cp[:, :-1]
            topki = topki[:, :-1, :]
            topkp = topkp[:, :-1, :]
        cy = cy - 1  # As input of LM where 0 is for blank and characters start from 1
        topki = topki - 2  # As index of character in the alphabet list
        for i_sample, pi in enumerate(cp):
            tgt_idx = self.target_sort(pi)
            if tgt_idx is None:
                break
            tgt_idx = tgt_idx.tolist()
            for idx in iter(tgt_idx):
                x_candidate = self.p_drop(cy[i_sample], cp[i_sample], idx)
                scores = self.lm(x_candidate.unsqueeze(0), torch.tensor([idx]).long().cuda())  # [1, nc]
                score_k = self.choose(scores, topki[i_sample, idx], topkp[i_sample, idx])  # [k]
                pii, yii = torch.max(score_k, 0)
                # if pii[0] > cp[i_sample, idx]:
                cy[i_sample, idx] = topki[i_sample, idx, yii.item()] + 1
                cp[i_sample, idx] = pii.item() * 1.8
                # cp[i_sample, idx] = 0.99
        cy = cy + 1
        return cy


def main(encoder, decoder, lm, device, label_map):
    test_loader = get_loader(is_training=False, alphabet=alphabet, bs=1)
    f_results = open('pred_refiner2.txt', 'w')
    cnt_former = 0
    cnt_true = 0
    refiner = RefineText2(lm)
    for batch_idx, (imgs, (targets, targets_len), idx) in enumerate(test_loader):
        imgs, targets, targets_len = imgs.to(device), targets.to(device), targets_len.to(device)
        targets = targets.permute(1, 0)[:, :-1]    # [b, L]
        (predicted0, _, cp, _, topkp, topki), _ = batch_test(
            imgs, encoder, decoder, max_len=40, need_refine=True, topk=args.topk)  # [b, L+1], [b, L+1], [b, L+1, k]
        # if not targets.equal(predicted0[:, :-1]):
        #     print("checking...")
        predicted1 = refiner(predicted0[:, :-1], cp[:, :-1], topki[:, :-1], topkp[:, :-1])    # [b, L]
        pred_string0 = label_map.decode(predicted0[:, :-1].squeeze(0).cpu().numpy())
        pred_string1 = label_map.decode(predicted1.squeeze(0).cpu().numpy())
        gt_string = label_map.decode(targets.squeeze(0).cpu().numpy())
        if pred_string0 == gt_string:
            cnt_former += 1
        if pred_string1 == gt_string:
            cnt_true += 1
        record = 'pred_before: {0}  pred_now: {1}  gt: {2}'.format(pred_string0, pred_string1, gt_string)
        f_results.writelines(record+'\n')
        # if batch_idx % 1000 == 999:
        if pred_string0 != gt_string:
            print(record)
            # print(cp[:, :-1])
    f_results.close()
    print("Accr={}/{}={:.2%}, previous_accr={}/{}={:.2%}".format(
        cnt_true, batch_idx+1, cnt_true/(batch_idx+1), cnt_former, batch_idx+1, cnt_former/(batch_idx+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--idc", type=str, default="0")
    parser.add_argument('--ocr', type=str, default='logs/model-520000.pth')
    parser.add_argument('--nlp', type=str, default='/workspace/ccx/experiments/NLP/FillTheBlank/param/attention2.pkl')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--use_stn', action='store_true', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.idc
    device = torch.device('cuda:0')

    f_alphabet = "/home/dataset/TR/synth_cn/alphabet.json"
    alphabet = get_alphabet(f_alphabet)
    label_map = Ids2Str(alphabet)
    num_classes = label_map.num_classes

    # get model
    encoder, decoder, lm = load_vanilla(num_classes, device, args)

    main(encoder, decoder, lm, device=device, label_map=label_map)
