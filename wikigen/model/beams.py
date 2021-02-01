import torch
from .utils import gather_last, mean_pooling, max_pooling


class Beam(object):
    def __init__(self, size, pad_id, bos_id, eos_id, cuda=True):

        self.size = size
        self.done = False
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.

        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad_id)]
        self.nextYs[0][0] = self.bos_id

        # The attentions (matrix) for each time.
        self.attn = []

        # The generation probabilities for each time
        self.p_gen = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, attnOut, pGenOut=None):

        # wordlk dimensions = beam x num words
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords

        self.prevKs.append(prevK)  # punteros al id del beam anterior
        self.nextYs.append(bestScoresId - prevK * numWords)
        self.attn.append(attnOut.index_select(0, prevK))

        if pGenOut is not None:
            self.p_gen.append(pGenOut.index_select(0, prevK))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos_id:
            self.done = True

        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        hyp, attn = [], []
        p_gen = []
        # print(len(self.prevKs), len(  self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            if self.p_gen:
                p_gen.append(self.p_gen[j][k])
            k = self.prevKs[j][k]

        if p_gen:
            p_gen = torch.stack(p_gen[::-1])

        return hyp[::-1], torch.stack(attn[::-1]), p_gen


def update_active(t, active_idx, num_active_beams, beam_size, batch_beam_index=0):
    # select only the remaining active sentences
    original_size = t.size()
    t = t.unsqueeze(batch_beam_index + 1)

    size = list(t.size())
    size[batch_beam_index] = num_active_beams
    size[batch_beam_index + 1] = beam_size
    t = t.view(*size)

    new_size = list(original_size)
    new_size[batch_beam_index] = (
        new_size[batch_beam_index] * len(active_idx) // num_active_beams
    )

    new_t = t.index_select(batch_beam_index, active_idx)
    new_t = new_t.view(*new_size)

    return new_t


def beam_search_decode(
    model,
    vocab,
    max_length,
    beam_size,
    context_batch_mask,
    context_batch,
    z,
    init_state,
):

    batch_size = z.shape[0]

    src_seq_len = context_batch_mask.size(1)
    # -> (batch_size*beam_size, src_seq_len, hidden_size)

    beam_context_batch = context_batch.unsqueeze(1)
    beam_context_batch = beam_context_batch.repeat(1, beam_size, 1, 1)
    beam_context_batch = beam_context_batch.view(
        batch_size * beam_size, src_seq_len, -1
    )

    beam_edit_repr = z.unsqueeze(1)
    beam_edit_repr = beam_edit_repr.repeat(1, beam_size, 1)
    beam_edit_repr = beam_edit_repr.view(batch_size * beam_size, -1)

    # -> (batch_size*beam_size, src_seq_len)
    beam_context_batch_mask = context_batch_mask.unsqueeze(1)
    beam_context_batch_mask = beam_context_batch_mask.repeat(1, beam_size, 1)
    beam_context_batch_mask = beam_context_batch_mask.view(
        batch_size * beam_size, src_seq_len
    )

    beam_init_state = init_state.unsqueeze(1)
    beam_init_state = beam_init_state.repeat(1, beam_size, 1)
    beam_init_state = beam_init_state.view(1, batch_size * beam_size, -1)

    beam_decoder_hidden_tuple_i = (
        beam_init_state,
        beam_init_state,
    )

    beams = [
        Beam(beam_size, vocab.tgt.PAD.hash, vocab.tgt.BOS.hash, vocab.tgt.EOS.hash,)
        for k in range(batch_size)
    ]

    num_active = batch_size
    batch_idx = range(batch_size)

    model.attention = []

    for i in range(max_length):

        # -> (num_active, beam_size)
        beam_tgt_batch_sequences_i = torch.stack(
            [b.get_current_state() for b in beams if not b.done]
        )

        # -> (num_active*beam_size, 1)
        beam_tgt_batch_sequences_i = beam_tgt_batch_sequences_i.view(
            num_active * beam_size, 1
        )

        beam_decoder_hidden_tuple_i, beam_logits_i = model.decoder.generate(
            beam_tgt_batch_sequences_i,
            beam_decoder_hidden_tuple_i,
            beam_context_batch,
            beam_context_batch_mask,
            beam_edit_repr,
        )

        beam_att_i = model.decoder.attention[-1]

        # (num_active*beam_size, output_size)
        beam_logits_i = torch.nn.functional.log_softmax(beam_logits_i, 1)

        # -> (num_active, beam_size, vocab_size)
        beam_logits_i = beam_logits_i.view(num_active, beam_size, -1).contiguous()

        # -> (num_active, beam_size, src_seq_len)
        beam_attentions_i = beam_att_i.view(num_active, beam_size, -1).contiguous()

        active = []

        for b in range(batch_size):

            if beams[b].done:
                continue

            idx = batch_idx[b]

            if not beams[b].advance(beam_logits_i[idx], beam_attentions_i[idx]):
                active += [b]

            # we update decoder hidden tuple with the states of the beams
            for beam_decoder_hidden in beam_decoder_hidden_tuple_i:
                # we  choose the segment corresponding to this beam for this item in the batch
                # -> (batch_size, beam_size, hidden_size)
                beam_decoder_hidden = beam_decoder_hidden.view(
                    1, num_active, beam_size, -1
                )
                # -> (1, beam_size, hidden_size)
                beam_decoder_hidden_i = beam_decoder_hidden[:, idx, :, :]

                # we in-place replace the data
                beam_decoder_hidden_i.copy_(
                    beam_decoder_hidden_i.index_select(1, beams[b].get_current_origin())
                )

        if not active:
            break

        # in this section, the sentences that are still active are
        # trimmed so that the decoder is not run on completed sentences

        active_idx = torch.LongTensor([batch_idx[k] for k in active]).cuda()
        batch_idx = {beam: idx for idx, beam in enumerate(active)}

        # -> (1, num_active*beam_size, hidden_size)
        beam_decoder_hidden_tuple_i = (
            update_active(
                beam_decoder_hidden_tuple_i[0],
                active_idx,
                num_active,
                beam_size,
                batch_beam_index=1,
            ),
            update_active(
                beam_decoder_hidden_tuple_i[1],
                active_idx,
                num_active,
                beam_size,
                batch_beam_index=1,
            ),
        )

        # -> (num_active*beam_size, src_seq_len, hidden_size)
        beam_context_batch = update_active(
            beam_context_batch, active_idx, num_active, beam_size, batch_beam_index=0
        )

        # -> (num_active*beam_size, src_seq_len)
        beam_context_batch_mask = update_active(
            beam_context_batch_mask,
            active_idx,
            num_active,
            beam_size,
            batch_beam_index=0,
        )
        beam_edit_repr = update_active(
            beam_edit_repr, active_idx, num_active, beam_size, batch_beam_index=0
        )
        num_active = len(active)

    # (4) package everything up
    n_best = 1
    allHyp, allScores, allAttn = [], [], []

    for b in range(batch_size):
        scores, ks = beams[b].sort_best()
        allScores += [scores[:n_best]]
        hyps, attn, p_gen = zip(*[beams[b].get_hyp(k) for k in ks[:n_best]])
        allAttn += [attn]
        allHyp += ([i.item() for i in hyps[0]],)

    return allHyp, torch.exp(torch.tensor(allScores)).mean()
