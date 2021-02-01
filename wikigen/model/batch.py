from itertools import chain
import numpy as np
import torch


class Batch:
    def __init__(self, sequences, lengths, masks, classes, tag_sequences):
        self.sequences = sequences
        self.lengths = lengths
        self.masks = masks
        self.classes = classes
        self.tag_sequences = tag_sequences

    def to_torch(self, device, fasttext=False):
        self.fasttext = fasttext
        if self.sequences is not None:
            if self.fasttext:
                self.sequences = self.sequences
                print(len(self.sequences))
            else:
                self.sequences = torch.tensor(
                    self.sequences, device=device, dtype=torch.long
                )

        if self.masks is not None:
            self.masks = torch.tensor(
                self.masks, device=device, dtype=torch.float
            )

        if self.lengths is not None:
            self.lengths = torch.tensor(
                self.lengths, device=device, dtype=torch.long
            )

        if self.classes is not None:
            self.classes = torch.tensor(
                self.classes, device=device, dtype=torch.float
            )

        if self.tag_sequences is not None:
            self.tag_sequences = torch.tensor(
                self.tag_sequences, device=device, dtype=torch.long
            )


def pad1d(
    sequences,
    dim0_pad=None,
    dim1_pad=None,
    pad_lengths=False,
    align_right=False,
    pad_id=0,
    dtype=None,
):
    """Pad a batch containing "1d" sequences
       Receive a list of sequences and return a padded 2d numpy ndarray,
       a numpy ndarray of lengths and a padded mask
       sequences: a list of lists, corresponding to sequences encoded in 1
                  hierarchical level, e.g. a sentence represented as a
                  sequence of words. The input `sequences` is a batch of such
                  sequences.
       len(sequences) = M, and N is the max sequence length contained in
       sequences.
        e.g.: [[2,45,3,23,54], [12,4,2,2], [4], [45, 12]]
       Return a numpy ndarray of dimension (M, N) corresponding to the
       padded sequence, a ndarray of the original lengths, and a mask.
       Returns:
           out: a numpy ndarray of dimension (M, N)
           lengths: a numpy ndarray of ints containing the lengths of each
                    input_list element
           mask: a numpy ndarray of dimension (M, N)
       """
    if not dim0_pad:
        dim0_pad = len(sequences)
    if not dim1_pad:
        dim1_pad = max(len(seq) for seq in sequences)
    if dtype == "str":
        out = np.full(
            shape=(dim0_pad, dim1_pad), fill_value=pad_id, dtype="<U50"
        )
    else:
        out = np.full(shape=(dim0_pad, dim1_pad), fill_value=pad_id)

    mask = np.zeros(shape=(dim0_pad, dim1_pad))

    lengths = []
    for i in range(len(sequences)):
        data_length = len(sequences[i])
        ones = np.ones(data_length)
        lengths.append(data_length)
        offset = dim1_pad - data_length if align_right else 0
        np.put(out[i], range(offset, offset + data_length), sequences[i])
        np.put(
            mask[i],
            range(offset, offset + data_length),
            np.ones(shape=(data_length)),
        )

    lengths = np.array(lengths)
    return out, lengths, mask


def pad2d(sequences2d, batch_first=True, pad_id=0):
    """Pad a batch containing "2d" sequences
       sequences2d: A list containing lists of lists, corresponding to
           sequences encoded in 2 hierarchical levels, e.g. a sentence
           represented as a sequence of words represented as sequences of
           characters. The input `sequences2d` is a batch of such
           sequences.
       e.g.: [
               [[1, 2, 3], [4, 5, 6]],
               [[7, 8, 9, 10, 11], [1, 2, 5, 3, 6]],
             ]
       return (padded_batch, first_h_lengths, second_h_lengths, masks) where
           padded_batch: 3d ndarray of dimension
                         (batch_size, max_sent_len, max_word_len)
           first_h_lengths: First hierarchy lengths. 1d ndarray of
                            dim (batch_size) corresponding to the first-level
                            hierarchy, e.g. sentence lengths
           second_h_lengths:second hierarchy lengths. 2d ndarray of
                            dim (batch_size, max_sent_len) corresponding to the
                            second level hierarchy, e.g. word lengths. Note
                            that this ndarray is padded with fake lengths of 1.
           masks: 3d ndarray with the same dim as padded_batch. This ndarrays
                  shows the positions of the valid items as opposed to the
                  paddings
       """
    if not batch_first:
        raise NotImplementedError

    batch_size = len(sequences2d)

    # TODO: rename variables make method more generic
    # max word length for the whole batch
    max_word_len = max(
        [max(len(word) for word in char_sent) for char_sent in sequences2d]
    )

    max_sent_len = max([len(char_sent) for char_sent in sequences2d])

    # The second hierarchy lengths are padded with ones because a length of 0
    # makes no sense later in the process
    second_h_lengths = np.ones(shape=(batch_size, max_sent_len), dtype=np.int64)

    padded_batch = []
    first_h_lengths = []
    masks = []
    for i, sequence in enumerate(sequences2d):
        padded_sent, word_lengths, mask = pad1d(
            sequence,
            dim0_pad=max_sent_len,
            dim1_pad=max_word_len,
            pad_id=pad_id,
        )

        np.put(second_h_lengths[i], range(len(word_lengths)), word_lengths)
        padded_batch.append(padded_sent)
        first_h_lengths.append(len(sequence))
        masks.append(mask)

    # -> (batch_size, max_sent_len, max_word_len)
    padded_batch = np.array(padded_batch)

    # -> (batch_size)
    first_h_lengths = np.array(first_h_lengths, dtype=np.int64)
    masks = np.array(masks)

    return padded_batch, first_h_lengths, second_h_lengths, masks


def categories_to_vector(no):
    out_list = np.zeros(4).tolist()
    out_list[no] = 1
    return out_list


class BatchIterator(object):
    def __init__(
        self,
        vocab,
        examples,
        batch_size,
        batch_builder,
        shuffle=False,
        max_len=-1,
        elmo=False,
    ):

        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.examples = examples
        self.num_batches = (len(self.examples) + batch_size - 1) // batch_size
        self.batch_builder = batch_builder
        self.elmo = elmo

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        examples_slice = []
        for i, example in enumerate(self.examples, 1):
            examples_slice.append(example)
            if i > 0 and i % (self.batch_size) == 0:
                yield self.batch_builder(
                    examples_slice,
                    self.vocab,
                    max_len=self.max_len,
                    elmo=self.elmo,
                )
                examples_slice = []

        if examples_slice:
            yield self.batch_builder(
                examples_slice,
                self.vocab,
                max_len=self.max_len,
                elmo=self.elmo,
            )


class YinEditBatch(object):
    def __init__(self, examples, vocab, max_len=None, elmo=False):

        self.examples = examples
        self.ids_batch = [example["id"] for example in examples]

        src_examples = []
        for example in examples:
            src_example_i = vocab.src.tokens2indices(
                example["src"], add_eos=True, add_bos=True
            )[:max_len]
            src_examples.append(src_example_i)

        src_padded, src_lengths, src_masks = pad1d(
            src_examples, pad_id=vocab.src.PAD.hash,
        )

        # src_examples = [vocab.src.tokens2indices(example['src'][:max_len])
        #                for example in examples]

        self.tgt_input_batch = None
        if "tgt" in examples[0]:
            tgt_input_examples = [
                vocab.tgt.tokens2indices(
                    example["tgt"], add_eos=True, add_bos=True
                )[:max_len]
                for example in examples
            ]

            tgt_input_padded, tgt_input_lengths, tgt_input_masks = pad1d(
                tgt_input_examples, pad_id=vocab.tgt.PAD.hash,
            )

            self.tgt_input_batch = Batch(
                tgt_input_padded, tgt_input_lengths, tgt_input_masks, None, None
            )

        self.changed = None
        if "changed" in examples[0]:
            changed_examples = [
                vocab.tgt.tokens2indices(
                    example["changed"], add_eos=False, add_bos=False
                )[:max_len]
                for example in examples
            ]

            changed_padded, changed_lengths, changed_masks = pad1d(
                changed_examples, pad_id=vocab.tgt.PAD.hash,
            )

            self.changed = Batch(
                changed_padded, changed_lengths, changed_masks, None, None
            )

        self.yin_before = None
        yin_before = None
        if "yin_before" in examples[0]:
            yin_before = [
                vocab.yin_before.tokens2indices(example["yin_before"][:max_len])
                for example in examples
            ]
            yin_before_padded, yin_before_lengths, yin_before_masks = pad1d(
                yin_before, pad_id=vocab.yin_before.PAD.hash
            )

            self.yin_before = Batch(
                yin_before_padded,
                yin_before_lengths,
                yin_before_masks,
                None,
                None,
            )

        yin_after = None
        self.yin_after = None
        if "yin_after" in examples[0]:
            yin_after = [
                vocab.yin_after.tokens2indices(example["yin_after"][:max_len])
                for example in examples
            ]
            assert [len(i) for i in yin_after] == [len(i) for i in yin_before]
            yin_after_padded, yin_after_lengths, yin_after_masks = pad1d(
                yin_after, pad_id=vocab.yin_after.PAD.hash
            )

            self.yin_after = Batch(
                yin_after_padded, yin_after_lengths, yin_after_masks, None, None
            )

        src_tag_padded = None
        if examples[0]["src_tag"][0]:
            src_tag_examples = [
                vocab.src_tag.tokens2indices(example["src_tag"][:max_len])
                for example in examples
            ]
            assert [len(i) for i in yin_after] == [
                len(i) for i in src_tag_examples
            ]

            src_tag_padded, _, _ = pad1d(
                src_tag_examples, pad_id=vocab.src_tag.PAD.hash
            )

        self.src_batch = Batch(
            src_padded, src_lengths, src_masks, None, src_tag_padded
        )

        self.added_batch = None
        self.removed_batch = None
        if vocab.all is not None:
            added_tokens_list = []
            removed_tokens_list = []
            for ex in self.examples:
                added_tokens = list(
                    set(ex["tgt"][:max_len]) - set(ex["src"][:max_len])
                )
                removed_tokens = list(
                    set(ex["src"][:max_len]) - set(ex["tgt"][:max_len])
                )
                if not added_tokens:
                    added_tokens = ["<BLANK>"]
                if not removed_tokens:
                    removed_tokens = ["<BLANK>"]

                added_tokens = vocab.all.tokens2indices(added_tokens)
                removed_tokens = vocab.all.tokens2indices(removed_tokens)
                added_tokens_list.append(added_tokens)
                removed_tokens_list.append(removed_tokens)

            added_padded, added_lengths, added_masks = pad1d(
                added_tokens_list, pad_id=vocab.all.PAD.hash
            )

            rem_padded, rem_lengths, rem_masks = pad1d(
                removed_tokens_list, pad_id=vocab.all.PAD.hash
            )

            self.added_batch = Batch(
                added_padded, added_lengths, added_masks, None, None
            )
            self.removed_batch = Batch(
                rem_padded, rem_lengths, rem_masks, None, None
            )

    def to_torch(self, device):
        self.src_batch.to_torch(device)

        if self.tgt_input_batch is not None:
            self.tgt_input_batch.to_torch(device)

        if self.yin_before is not None:
            self.yin_before.to_torch(device)

        if self.yin_after is not None:
            self.yin_after.to_torch(device)

        if self.changed is not None:
            self.changed.to_torch(device)

        if self.added_batch is not None:
            self.added_batch.to_torch(device)

        if self.removed_batch is not None:
            self.removed_batch.to_torch(device)
