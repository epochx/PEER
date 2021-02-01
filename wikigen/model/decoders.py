import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import random
from .attention import AttentionLayer


class LSTMVAEDecoder(nn.Module):
    def __init__(self, input_dropout, output_dropout, hidden_size):

        self.decoder = nn.LSTM()


class EditDecoder(nn.Module):
    def __init__(
        self,
        embeddings,
        hidden,
        repr_hidden_size,
        num_layers,
        input_dropout=0,
        output_dropout=0,
        word_dropout=1.0,
        teacher_forcing_p=0.5,
        attention="dot",
    ):

        super(EditDecoder, self).__init__()

        self.embeddings = embeddings

        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.repr_hidden_size = repr_hidden_size

        self.hidden_size = hidden
        self.word_dropout = word_dropout
        self.decoder = nn.LSTM(
            self.embeddings.embedding_dim + self.repr_hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.teacher_forcing_p = teacher_forcing_p

        self.attention_layer = AttentionLayer(
            self.hidden_size, scoring_scheme=attention
        )

        # The 2 appears because we will concatenate the decoded vector with the
        # attended decoded vector
        num_catted_vectors = 2

        self.composer_layer = nn.Linear(
            self.hidden_size * num_catted_vectors, self.hidden_size
        )

        self.output_layer = nn.Linear(self.hidden_size, self.embeddings.num_embeddings)

        self.loss_function = nn.CrossEntropyLoss(
            size_average=False, ignore_index=self.embeddings.padding_idx
        )

    def forward(
        self,
        representation,
        encoder_hidden_states,
        seq,
        src_batch_mask,
        initial_state=None,
    ):
        class_name = self.__class__.__name__

        assert representation.shape == torch.Size([seq.shape[0], self.repr_hidden_size])

        logits = []
        predictions = []
        attention = []
        batch_size, seq_len = seq.shape
        # batch_size, 1
        original_seq = seq
        seq_i = seq[:, 0].unsqueeze(1)
        if self.training:
            word_dropout = Bernoulli(self.word_dropout).sample(seq[:, 1:].shape)
            word_dropout = word_dropout.type(torch.LongTensor)
            seq = seq.cpu()
            seq[:, 1:] = seq[:, 1:] * word_dropout
            seq = seq.cuda()

        # 1, batch_size, hidden_x_dirs
        if initial_state is not None:
            decoder_hidden_tuple_i = (
                initial_state.unsqueeze(0),
                initial_state.unsqueeze(0),
            )
        else:
            decoder_hidden_tuple_i = None
        # teacher forcing p
        p = random.random()

        self.attention = []

        # we skip the EOS as input for the decoder
        for i in range(seq_len - 1):

            decoder_hidden_tuple_i, logits_i = self.generate(
                seq_i,
                decoder_hidden_tuple_i,
                encoder_hidden_states,
                src_batch_mask,
                representation,
            )

            # batch_size
            _, predictions_i = logits_i.max(1)

            logits.append(logits_i)
            predictions.append(predictions_i)

            if self.training and p <= self.teacher_forcing_p:
                # batch_size, 1
                seq_i = seq[:, i + 1].unsqueeze(1)
            else:
                # batch_size, 1
                seq_i = predictions_i.unsqueeze(1)
                seq_i = seq_i.cuda()

        # (seq_len, batch_size)
        predictions = torch.stack(predictions, 0)

        # (batch_size, seq_len)
        predictions = predictions.t().contiguous()

        # (seq_len, batch_size, output_size)
        logits = torch.stack(logits, 0)

        # (batch_size, seq_len, output_size)
        logits = logits.transpose(0, 1).contiguous()

        # (batch_size*seq_len, output_size)
        flat_logits = logits.view(batch_size * (seq_len - 1), -1)

        # (batch_size, seq_len)
        labels = original_seq[:, 1:].contiguous()

        # (batch_size*seq_len)
        flat_labels = labels.view(-1)

        loss = self.loss_function(flat_logits, flat_labels)

        ppl = torch.exp(
            F.cross_entropy(
                flat_logits, flat_labels, ignore_index=self.embeddings.padding_idx
            )
        )
        return loss, predictions, ppl

    def generate(
        self,
        tgt_batch_sequences_i,
        decoder_hidden_tuple_i,
        encoder_hidden_states,
        src_batch_mask,
        representation,
    ):
        """

        :param tgt_batch_i: torch.LongTensor(1, batch_size)
        :param decoder_hidden_tuple_i: tuple(torch.FloatTensor(1, batch_size, hidden_size))
        :param encoder_hidden_states: torch.FloatTensor(batch_size, seq_len, hidden_x_dirs)
        :param src_batch_mask: torch.LongTensor(batch_size, seq_len)
        :param comment_hidden_states: ?
        :param com_batch_mask: ?
        :return:
        """

        # (batch_size, 1, embedding_size)
        emb_tgt_batch_i = self.embeddings(tgt_batch_sequences_i)
        emb_tgt_batch_i = torch.cat([emb_tgt_batch_i, representation.unsqueeze(1)], -1)
        emb_tgt_batch_i = self.input_dropout(emb_tgt_batch_i)

        # (batch_size, 1, hidden_x_dirs) and (1, batch_size, hidden_size)
        decoder_hidden_states_i, decoder_hidden_tuple_i = self.decoder(
            emb_tgt_batch_i, decoder_hidden_tuple_i
        )

        # batch_size, hidden_x_dirs
        s_i = decoder_hidden_states_i.squeeze(1)

        # (batch_size, hidden_x_dirs) and  (batch_size, seq_len)
        t_i, attn_i = self.attention_layer.forward(
            encoder_hidden_states, s_i, src_batch_mask
        )

        self.attention.append(attn_i)

        # batch_size, 2*hidden_x_dirs
        s_t_i = torch.cat([s_i, t_i], 1)

        # batch_size, hidden_x_dirs
        new_s_i = self.composer_layer(s_t_i)
        new_s_i = self.output_dropout(new_s_i)

        # batch_size, output_size
        logits_i = self.output_layer(new_s_i)

        return decoder_hidden_tuple_i, logits_i
