from __future__ import absolute_import
import errno
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import os
import sys


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sequence_mask(seq_len, max_len=None):
    if max_len is None:
        max_len = seq_len.data.max()
    batch_size = seq_len.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).cuda()
    seq_length_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
    mask = seq_range_expand < seq_length_expand
    return mask


def masked_cross_entropy(logits, target, length):
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    losses = losses_flat.view(*target.size())
    mask = sequence_mask(length, target.size(1))

    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()

    pred_flat = log_probs_flat.max(1)[1]

    pred_seqs = pred_flat.view(*target.size()).transpose(0, 1).contiguous()
    mask_flat = mask.view(-1)

    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words


def save_state(ckpt_dir, step, encoder, decoder, optimizers):
    save_state_dict = {
        "step": step,
        "state_dict": {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": [x.state_dict() for x in optimizers],
        },
    }
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_filepath = os.path.join(ckpt_dir, "model-{}.pth".format(step))
    torch.save(save_state_dict, save_filepath)
    print("=> saving checkpoint at %s." % save_filepath)


def load_state(ckpt_dir, step, encoder, decoder, optimizers=None):
    save_filepath = os.path.join(ckpt_dir, "model-%d.pth" % step)
    if os.path.isfile(save_filepath):
        checkpoint = torch.load(save_filepath)
        encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
        decoder.load_state_dict(checkpoint["state_dict"]["decoder"])
        if optimizers is not None:
            state_dicts = checkpoint["state_dict"]["optimizer"]
            for optimizer, state_dict in zip(optimizers, state_dicts):
                try:
                    optimizer.load_state_dict(state_dict)
                except:
                    print('Cannot match the model params completely --- strict=False')
                    optimizer.load_state_dict(state_dict, strict=False)
        print("=> restore checkpoint from %s finished." % save_filepath)
    else:
        print("=> no checkpoint found at %s." % save_filepath)
