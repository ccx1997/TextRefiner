import torch
import torch.nn as nn


class SequenceCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits, targets, target_lens):
        """
        Args: 
            logits, torch.cuda.FloatTensor [minibatch, C, d],
                which is just the plain output from the RNN decoder, not passed the softmax func.
            targets, torch.cuda.LongTensor [minibatch, d]
            target_lens: torch.cuda.LongTensor [minibatch]
        """
        losses = self.cross_entropy(logits, targets)
        batch_size, max_len = targets.size()
        seq_range = torch.arange(0, max_len).long().unsqueeze(0).expand(
                batch_size, max_len).cuda()
        seq_len = target_lens.unsqueeze(1).expand_as(seq_range)
        mask = (seq_range < seq_len).float()
        losses = losses * mask
        loss = losses.sum() / target_lens.float().sum()
        return loss
