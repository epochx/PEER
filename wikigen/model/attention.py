import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as weight_init


def concat_scoring(hidden_states, sent_repr, W, context):
    """
    Args:
        hidden_states: (batch_size, seq_len, hidden_x_dirs)
        sent_repr: (batch_size, seq_len, hidden_x_dirs),
        W: parameter used for concat and general attention types;
           dim(W): (2 * hidden_x_dirs, hidden_x_dirs)
        context: (hidden_x_dirs, 1)"""

    # -> (batch_size, seq_len, 2 * hidden_x_dirs)
    concat_sent_batch = torch.cat([hidden_states, sent_repr], 2)

    # -> (batch_size, seq_len, hidden_x_dirs)
    WY_plus_WR = torch.bmm(concat_sent_batch, W)
    # dim(M) = (batch_size, seq_len, hidden_x_dirs)
    M = F.tanh(WY_plus_WR)

    # -> (batch_size, seq_len, 1)
    score = torch.bmm(M, context)

    return score


def dot_scoring(hidden_states, sent_repr):
    """
    Calculate the dot product between every column of both tensors
    Args:
        hidden_states: (batch_size, seq_len, hidden_x_dirs)
        sent_repr: (batch_size, seq_len, hidden_x_dirs)"""

    # -> = (batch_size, seq_len, hidden_x_dirs)
    H = torch.mul(hidden_states, sent_repr)

    # -> = (batch_size, seq_len, 1)
    score = torch.sum(H, 2)

    return score


def general_scoring(hidden_states, sent_repr, W):
    """Args:
           hidden_states: (batch_size, seq_len, hidden_x_dirs)
           sent_repr: (batch_size, 1, hidden_x_dirs)
           W: (batch_size, hidden_x_dirs, hidden_x_dirs)"""

    # -> (batch_size, seq_len, hidden_x_dirs)
    H_W = torch.bmm(hidden_states, W)
    # -> (batch_size, seq_len, 1)
    score = torch.bmm(H_W, sent_repr.transpose(1, 2))

    return score


class ScoringLayer(nn.Module):
    def __init__(self, input_size, att_type="concat"):
        """
        Different scoring schemes for calculating the attentions;
        See "Effective Approaches to Attention-based Neural Machine
        Translation"
        https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
        Args:
           hidden_states: (batch_size, seq_len, hidden_x_dirs)
           sent_repr: (batch_size, hidden_x_dirs),
           W: parameter used for concat and general attention types;
              dim(W): (batch_size, *, hidden_x_dirs)
           context: (hidden_x_dirs)"""
        super(ScoringLayer, self).__init__()
        self.att_type = att_type
        self.input_size = input_size

        if self.att_type == "concat":
            # self.W is W^y in the paper
            self.W = Parameter(torch.Tensor(2 * self.input_size, self.input_size))
            # self.context is w in the paper
            self.context = Parameter(torch.Tensor(self.input_size))

        elif self.att_type == "general":
            self.W = Parameter(torch.Tensor(self.input_size, self.input_size))
        else:
            self.W = None

        self.reset_parameters()

    def reset_parameters(self):
        """This initialization is overriden if the param_init parameter
           is provided to the run file"""
        if self.W is not None:
            tanh_gain = weight_init.calculate_gain("tanh")
            weight_init.xavier_normal_(self.W, tanh_gain)
            # self.W.data.uniform_(-0.001, 0.001)

    def forward(self, hidden_states, sent_repr):
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        hidden_x_dirs = hidden_states.size(2)

        # -> (batch_size, 1, hidden_x_dirs)
        sent_repr_exp = sent_repr.unsqueeze(1)

        # -> (batch_size, seq_len, hidden_x_dirs)
        sent_repr_exp = sent_repr_exp.repeat(1, seq_len, 1)
        if self.att_type == "concat":
            W = self.W.unsqueeze(0).expand(batch_size, 2 * hidden_x_dirs, hidden_x_dirs)
            # -> (batch_size, hidden_x_dirs)
            context = self.context.unsqueeze(0).expand(batch_size, hidden_x_dirs)
            # -> (batch_size, hidden_x_dirs, 1)
            context = context.unsqueeze(2)
            return concat_scoring(hidden_states, sent_repr_exp, W, context)

        elif self.att_type == "dot":
            return dot_scoring(hidden_states, sent_repr_exp)

        elif self.att_type == "general":
            W = self.W.unsqueeze(0).expand(batch_size, hidden_x_dirs, hidden_x_dirs)
            # -> (batch_size, 1, hidden_x_dirs)
            sent_repr = sent_repr.unsqueeze(1)
            return general_scoring(hidden_states, sent_repr, W)

        else:
            raise NotImplementedError

    def __repr__(self):
        s = "{name}("
        s += "att_type={att_type}"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class AttentionLayer(nn.Module):
    """Word-level attention as seen in https://arxiv.org/pdf/1605.09090.pdf
    Linear part inspired on
    http://pytorch.org/docs/_modules/torch/nn/modules/linear.html#Linear
    """

    def __init__(self, input_size, scoring_scheme="dot"):
        """input_size should be hidden_x_dirs"""
        super(AttentionLayer, self).__init__()
        self.input_size = input_size

        self.scoring_layer = ScoringLayer(self.input_size, att_type=scoring_scheme)

    def forward(self, sent_batch, mean_sent_batch, batch_mask):
        """
        dim(sent_batch) = (batch_size, seq_len, hidden_x_dirs)
        dim(mean_sent_batch) = (batch_size, hidden_x_dirs)
        dim(batch_mask) = (batch_size, seq_len)
        dim(self.W) = (hidden_x_dirs, hidden_x_dirs)
        """

        batch_size = sent_batch.size(0)
        seq_len = sent_batch.size(1)
        hidden_x_dirs = sent_batch.size(2)
        assert batch_size == mean_sent_batch.size(0)
        assert self.input_size == hidden_x_dirs
        assert hidden_x_dirs == mean_sent_batch.size(1)

        # -> (seq_len, batch_size, hidden_x_dirs)
        # batch_mask = batch_mask.unsqueeze(2).expand_as(sent_batch)
        # Make padding values = 0
        # r_sent_batch = torch.mul(batch_mask, sent_batch)

        # -> (batch_size, seq_len, hidden_x_dirs)
        # r_sent_batch = r_sent_batch.transpose(0, 1).contiguous()

        att_mask = 1 - batch_mask

        att_mask = att_mask.bool()
        # -> (batch_size, seq_len, 1); corresponds to u in the paper
        score = self.scoring_layer(sent_batch, mean_sent_batch)

        # -> (batch_size, seq_len)
        if len(score.size()) == 3:
            score = score.squeeze(2)

        # -> (batch_size, seq_len)
        score = score.masked_fill(att_mask, -float("inf"))

        # Since the input to softmax is 2D, then the operation is performed in
        # dim 1 (seq_len).
        # -> (batch_size, seq_len)
        alpha = F.softmax(score, 1)

        # -> (batch_size, 1, seq_len)
        r_alpha = alpha.unsqueeze(1)

        # -> (batch_size, 1, hidden_x_dirs)
        out = torch.bmm(r_alpha, sent_batch)

        # -> (batch_size, hidden_x_dirs)
        out = out.squeeze(1)

        return out, alpha
