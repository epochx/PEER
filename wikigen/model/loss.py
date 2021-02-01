import torch
import torch.nn as nn
import torch.nn.functional as F


class BoWLoss(nn.Module):
    def __init__(self, latent_size, vocab_size):

        super(BoWLoss, self).__init__()
        self.linear = nn.Linear(latent_size, vocab_size)
        self.log_softmax_fn = nn.LogSoftmax(dim=1)

    def forward(self, batch_latent, batch_labels, batch_labels_mask):
        batch_size, latent_size = batch_latent.shape

        batch_size_, seq_len = batch_labels.shape

        assert batch_size == batch_size_

        # -> batch_size, vocab_size
        batch_logits = self.linear(batch_latent)
        batch_log_probs = self.log_softmax_fn(batch_logits)

        # batch_size, seq_len
        batch_bow_lls = torch.gather(batch_log_probs, 1, batch_labels)

        masked_batch_bow_lls = batch_bow_lls * batch_labels_mask

        batch_bow_ll = masked_batch_bow_lls.sum(1)

        batch_bow_nll = -batch_bow_ll.sum()

        return batch_bow_nll
