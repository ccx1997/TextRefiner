import torch
import torch.nn.functional as F
import random
import os
import cv2
import numpy as np
from utils.utils import load_state

MAX_LENGTH = 40
SOS_token = 0
EOS_token = 1


def batch_train(input_tensor, target_tensor, target_len_tensor, encoder, decoder, criterion, teacher_forcing_ratio=0.5):
    batch_size = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs, recitified = encoder(input_tensor)
    decoder_input = torch.tensor([SOS_token] * batch_size).cuda()  # for <SOS>
    decoder_hidden = decoder.init_hidden(batch_size)
    logits = torch.zeros(target_length, batch_size, decoder.y_dim).cuda()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing:Feed the target at the next input
        for di in range(target_length):
            logits[di], decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  # Teacherforcing
    else:
        # Without teacher forcing use its own prediction as the next input
        all_ones = torch.ones(batch_size).cuda()
        all_finished = torch.zeros(batch_size).cuda()
        for di in range(target_length):
            logits[di], decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = logits[di].topk(1)
            decoder_input = topi.squeeze().detach()
            all_finished = torch.where(decoder_input == EOS_token, all_ones, all_finished).cuda()
            if all_finished.sum() == batch_size:
                break
    logits = logits.permute(1, 2, 0)  # [b, c, T]
    target_tensor = target_tensor.permute(1, 0)  # [b, T]
    loss = criterion(logits, target_tensor, target_len_tensor)
    loss.backward()
    return loss.item(), recitified


def batch_test(input_tensor, encoder, decoder, max_len=MAX_LENGTH, visualize=False, need_refine=False, topk=None):
    encoder.eval(), decoder.eval()
    with torch.no_grad():
        batch_size = input_tensor.size(0)
        encoder_outputs, recitified = encoder(input_tensor)
        decoder_input = torch.tensor([SOS_token] * batch_size).cuda()  # for <SOS>
        decoder_hidden = decoder.init_hidden(batch_size)
        outputs = []
        all_ones = torch.ones(batch_size).cuda()
        all_finished = torch.zeros(batch_size).cuda()
        attentions = []
        cfs = []
        probs = []
        topkvs = []
        topkis = []

        for di in range(max_len):
            output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                        need_refine=(need_refine or visualize))
            attentions.append(attention)
            if visualize or need_refine:
                output, cf = output
                output = F.softmax(output, dim=1)
                cfs.append(cf)
            topv, topi = output.topk(1)  # [b, 1]
            probs.append(topv.squeeze(-1))
            decoder_input = topi.squeeze(-1)
            outputs.append(decoder_input)
            if topk is not None:
                topkv, topki = output.topk(topk)  # [b, k]
                topkvs.append(topkv)
                topkis.append(topki)
            all_finished = torch.where(decoder_input == EOS_token, all_ones, all_finished).cuda()
            if all_finished.sum() == batch_size:
                break
        outputs = torch.stack(outputs, dim=1)  # [b,T]
        if visualize or need_refine:
            attentions = torch.stack(attentions, dim=1)  # [b, T, w]
            probs = torch.stack(probs, dim=1)  # [b,T]
            cfs = torch.stack(cfs, dim=1)  # [b, T, c]
            if topk is None:
                outputs = (outputs, attentions, probs, cfs)
            else:
                topkvs = torch.stack(topkvs, dim=1)  # [b, T, k]
                topkis = torch.stack(topkis, dim=1)  # [b, T, k]
                outputs = (outputs, attentions, probs, cfs, topkvs, topkis)
    return outputs, recitified


def evaluate(encoder, decoder, ckpt_dir, restore_iter,
             dataloader, label_map, accuracy_file, max_len=MAX_LENGTH):
    import string
    load_state(ckpt_dir, restore_iter, encoder, decoder)
    num_total = 0
    num_correct = 0
    writer = open(os.path.join(ckpt_dir, 'recogntion_results.txt'), 'w')

    def _normalize_text(text):
        text = "".join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    count = 0
    for i, (images, (targets, targets_len), _) in enumerate(dataloader):
        num_total += images.size(0)
        outputs, _ = batch_test(images.cuda(), encoder, decoder, max_len)
        outputs_str = label_map.decode(outputs.cpu().numpy())
        targets_str = label_map.decode(targets.permute(1, 0).numpy())
        for output_str, target_str in zip(outputs_str, targets_str):
            output_str = _normalize_text(output_str)
            target_str = _normalize_text(target_str)
            writer.write("%04d: %-20s %-20s\n" % (count, target_str, output_str))
            if output_str == target_str:
                num_correct += 1
            else:
                print("%04d: %-20s %-20s" % (count, target_str, output_str))
            count += 1
    print_buf = "=> iter: %d,  %d / %d = %.4f\n" % (
        restore_iter, num_correct, num_total, (num_correct / num_total))
    print(print_buf)
    with open(accuracy_file, "a") as f:
        f.write(print_buf)
    writer.close()


def truncate_string(raw_arr):
    """
    :param raw_arr: a np array consisting of index of characters, where negative represents the EoS
    :return: raw_arr[:id_of_EoS]
    """
    try:
        idx = np.where(raw_arr < 0)
        trunc = idx[0][0]
        result_ts = raw_arr[:trunc]
    except:
        result_ts = raw_arr
    return result_ts


class Ids2Str(object):
    """
    :param alphabet: a sorted list
    :return: a corresponding string
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.num_classes = len(self.alphabet) + 2

    def decode(self, ids, real_index=False):
        """
        :param ids: a numpy array of size [b, T] or [T,], containing index of characters
        :return a list of string(s)
        """
        if not real_index:
            ids = ids - 2
        if ids.ndim == 1:
            ids = truncate_string(ids)
            str_list = []
            for i in iter(ids):
                try:
                    str_list.append(self.alphabet[i] if i >= 0 else '[EoS]')
                except:
                    str_list.append('[NULL]')
            return [''.join(str_list)]
        else:
            strs = []
            for line in ids:
                strs.extend(self.decode(line, real_index=True))
            return strs


def visualize_attention(attention_arr, w, dh=12):
    """
    :param attention_arr: numpy array of size [L, w0]
    :param w: the width of image
    :param dh: height of every attention map line
    :return: array that can used to draw an image directly
    """
    h0, w0 = attention_arr.shape
    tmp = np.tile(attention_arr, (1, dh)).reshape(-1, w0)
    return cv2.resize(tmp, (w, h0 * dh))
