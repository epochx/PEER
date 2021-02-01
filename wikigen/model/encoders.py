import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import AttentionLayer
from wikigen.model.utils import gather_last
from torch.distributions.bernoulli import Bernoulli


class LSTM_VAEncoder(nn.Module):
    def __init__(
        self,
        before_edit_embeddings,
        after_edit_embeddings,
        tag_embeddings,
        hidden,
        num_layers,
        input_dropout=0.5,
        output_dropout=0.5,
    ):
        super(LSTM_VAEncoder, self).__init__()

        self.before_edit_embed = before_edit_embeddings
        self.tag_embed = tag_embeddings
        self.after_edit_embed = after_edit_embeddings
        self.hidden = hidden
        self.num_layers = num_layers
        self.encoder_input_size = sum(
            (
                self.before_edit_embed.embedding_dim,
                self.after_edit_embed.embedding_dim,
                self.tag_embed.embedding_dim,
            )
        )
        self.encoder = nn.LSTM(
            self.encoder_input_size,
            self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, before_edit, after_edit, tags, seq_lens):
        assert before_edit.shape == after_edit.shape
        assert before_edit.shape == tags.shape

        embedded_before = self.before_edit_embed(before_edit)
        embedded_after = self.after_edit_embed(after_edit)
        embedded_tags = self.tag_embed(tags)

        encoder_input = torch.cat(
            [embedded_before, embedded_tags, embedded_after], -1
        )
        encoder_input = self.input_dropout(encoder_input)
        encoder_input = nn.utils.rnn.pack_padded_sequence(
            encoder_input, seq_lens, batch_first=True, enforce_sorted=False
        )

        encoder_hidden, (h_0, c_0) = self.encoder(encoder_input)
        encoder_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            encoder_hidden, batch_first=True
        )
        encoder_hidden = self.output_dropout(encoder_hidden)

        final_hidden = encoder_hidden[
            torch.arange(encoder_hidden.size(0)), seq_lens - 1, :
        ]
        # TODO Highway layers

        return final_hidden, encoder_hidden


class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        embeddings,
        hidden,
        num_layers,
        input_dropout=0.5,
        output_dropout=0.5,
        word_dropout=1.0,
    ):
        super(LSTM_Encoder, self).__init__()

        self.embed = embeddings
        self.hidden = hidden
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            self.embed.embedding_dim,
            self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        self.word_dropout = word_dropout

    def forward(self, seq, seq_lens):

        if self.training:
            word_dropout = Bernoulli(self.word_dropout).sample(seq.shape)
            word_dropout = word_dropout.type(torch.LongTensor)
            seq = seq.cpu()
            seq = seq * word_dropout
            seq = seq.cuda()

        embedded_seq = self.embed(seq)
        embedded_seq = self.input_dropout(embedded_seq)

        encoder_input = nn.utils.rnn.pack_padded_sequence(
            embedded_seq, seq_lens, batch_first=True, enforce_sorted=False
        )

        encoder_hidden, (h_0, c_0) = self.encoder(encoder_input)
        encoder_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            encoder_hidden, batch_first=True
        )
        encoder_hidden = self.output_dropout(encoder_hidden)

        final_hidden = encoder_hidden[
            torch.arange(encoder_hidden.size(0)), seq_lens - 1, :
        ]

        # TODO Highway layers

        return final_hidden, encoder_hidden


class GuuEncoder(nn.Module):
    def __init__(
        self,
        embeddings,
        hidden_size,
        kappa,
        input_dropout=0.5,
        output_dropout=0.5,
        norm_eps=0.1,
        norm_max=14.0,
        use_kl=True,
    ):
        super(GuuEncoder, self).__init__()

        self.embed = embeddings
        self.linear_prenoise = nn.Linear(
            embeddings.embedding_dim, int(hidden_size / 2), bias=False
        )
        self.noise_scaler = kappa
        self.norm_eps = norm_eps
        self.norm_max = norm_max
        self.normclip = nn.Hardtanh(0, self.norm_max - norm_eps)
        self.use_kl = use_kl

    def forward(
        self, added_tokens_batch, removed_tokens_batch,
    ):

        embedded_added = self.embed(added_tokens_batch.sequences)
        embedded_removed = self.embed(removed_tokens_batch.sequences)

        # TODO: double-check the PADs are actually zero

        embedded_added_sum = embedded_added.sum(1)
        embedded_removed_sum = embedded_removed.sum(1)

        insert_set = self.linear_prenoise(embedded_added_sum)
        delete_set = self.linear_prenoise(embedded_removed_sum)

        combined_map = torch.cat([insert_set, delete_set], 1)
        if self.training and self.use_kl:
            combined_map = self.sample_vMF(combined_map, self.noise_scaler)

        return combined_map

    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        device = munorm.device
        trand = torch.rand(1).expand(munorm.size()) * eps
        return self.normclip(munorm) + trand.to(device)

    def sample_vMF(self, mu, kappa):
        """vMF sampler in pytorch.
        http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
        Args:
            mu (Tensor): of shape (batch_size, 2*word_dim)
            kappa (Float): controls dispersion. kappa of zero is no dispersion.
        """
        batch_size, id_dim = mu.size()
        result_list = []
        device = mu.device
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                # sample offset from center (on sphere) with spread kappa
                w = self._sample_weight(kappa, id_dim)
                wtorch = w.to(device) * torch.ones(id_dim).to(device)

                # sample a point v on the unit sphere that's orthogonal to mu
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)

                # compute new point
                scale_factr = torch.sqrt(
                    torch.ones(id_dim).to(device) - torch.pow(wtorch, 2)
                )
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munoise
            else:
                rand_draw = torch.randn(id_dim).to(device)
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(
                    id_dim
                )
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * rand_norms.to(device)  # mu[i]
            result_list.append(sampled_vec)

        return torch.stack(result_list, 0)

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (
            np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa
        )  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(
            1 - x ** 2
        )  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(
                dim / 2.0, dim / 2.0
            )  # concentrates towards 0.5 as d-> inf
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(
                u
            ):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return torch.FloatTensor((w,))

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = torch.randn(dim).to(device=mu.device)
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)


def test_sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    while True:
        z = np.random.beta(dim / 2.0, dim / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
            return torch.from_numpy(w)


def get_ev(kappa, dim, nsamp):
    samp_in = np.array([test_sample_weight(kappa, dim) for i in range(nsamp)])
    return (
        np.mean(samp_in),
        np.std(samp_in),
        np.percentile(samp_in, np.arange(0, 100, 10)),
    )


def get_mode(kappa, dim):
    return np.sqrt(4 * (kappa ** 2.0) + dim ** 2.0 + 6 * dim + 9) / (
        2 * kappa
    ) - (dim + 3.0) / (2 * kappa)

