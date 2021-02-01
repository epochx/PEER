#!/usr/bin/env python
# -*-coding: utf8 -*-
import time
import os
import json
import random
import hashlib
import numpy as np
import pickle
from itertools import chain
from collections import namedtuple, defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


from .settings import (
    SPLITS_PATH,
    EDITS_PATH,
)

from .vocab import Vocab

random.seed(42)


def max_idx(*args):
    return max(set(chain(*args)))


def load_stripped_json(line):
    return json.loads(line.strip())


def parse_jsonl(jsonl_path, max_len):

    jsonl_file = open(jsonl_path, "rb")
    indexes = list(range(max_len))

    with Pool(12) as p:
        dictionary = dict(
            zip(
                indexes,
                tqdm(
                    p.imap(load_stripped_json, jsonl_file), total=max(indexes)
                ),
            )
        )
        p.close()
    return dictionary


def split_list(examples, train_ratio=0.72, valid_ratio=0.08, test_ratio=0.2):
    """
    :param examples: list
    :param train_ratio: 0.8
    :return:
    """
    if test_ratio:
        assert (
            1.0 - 1e-8 <= train_ratio + valid_ratio + test_ratio <= 1.0 + 1e-8
        ), "ratios don't sum to 1"
    else:
        assert train_ratio + valid_ratio == 1.0, "ratios don't sum 1"

    num_examples = len(examples)
    random.shuffle(examples)

    last_train_idx = int(num_examples * train_ratio)
    last_valid_idx = int(last_train_idx + (num_examples * valid_ratio))

    train_data = examples[:last_train_idx]
    if test_ratio:
        valid_data = examples[last_train_idx:last_valid_idx]
        test_data = examples[last_valid_idx:]
        return train_data, valid_data, test_data
    else:
        valid_data = examples[last_train_idx:]
        return train_data, valid_data, valid_data


class ParsedEditsFile(object):
    def __init__(self, parsed_jsonl, indices=None):
        self.jsonl_file = parsed_jsonl
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield self.jsonl_file[idx]


VocabTuple = namedtuple(
    "VocabTuple", ["src", "src_tag", "tgt", "yin_before", "yin_after", "all"]
)


class AutoEncoderDataset(object):
    def __init__(
        self,
        save_path,
        name,
        tgt_min_freq,
        src_min_freq=1,
        joint=False,
        max_len=None,
        force_reload=False,
        lowercase=False,
        use_temporal_splits=False,
        generate=True,
    ):

        params = {
            key: value
            for key, value in locals().items()
            if key not in ["self", "save_path", "force_reload"]
        }

        self.hash = hashlib.sha1(str(params).encode("utf-8")).hexdigest()[:8]
        self.save_path = save_path
        self.name = name
        self.src_min_freq = src_min_freq
        self.tgt_min_freq = tgt_min_freq
        self.joint = joint
        self.generate = generate
        self.max_len = max_len
        self.lowercase = lowercase
        self.use_temporal_splits = use_temporal_splits
        self._read_files()

        self._read_or_build_vocabs(force_reload=force_reload)

    def _read_files(self):

        if self.use_temporal_splits:
            raise NotImplementedError

        else:
            base_indices_file_path = os.path.join(SPLITS_PATH, self.name)

        parsed_commits_path = os.path.join(EDITS_PATH, self.name + ".jsonl")

        train_indices_file_path = base_indices_file_path + ".train.txt"
        with open(train_indices_file_path) as f:
            train_indices = set(map(lambda x: int(x.strip()), f.readlines()))

        valid_indices_file_path = base_indices_file_path + ".valid.txt"
        with open(valid_indices_file_path) as f:
            valid_indices = set(map(lambda x: int(x.strip()), f.readlines()))

        test_indices_file_path = base_indices_file_path + ".test.txt"
        with open(test_indices_file_path) as f:
            test_indices = set(map(lambda x: int(x.strip()), f.readlines()))

        parsed_commits_path = parse_jsonl(
            parsed_commits_path,
            max_idx(train_indices, test_indices, valid_indices) + 1,
        )

        self.train = ParsedEditsFile(parsed_commits_path, train_indices)

        self.valid = ParsedEditsFile(parsed_commits_path, valid_indices)
        self.test = ParsedEditsFile(parsed_commits_path, test_indices)

    def _read_or_build_vocabs(self, force_reload=False):

        save_path = os.path.join(
            self.save_path, "{0}.{1}.data".format(self.name, self.hash)
        )

        if not os.path.exists(save_path) or force_reload:

            tgt_vocab = Vocab(
                add_padding=True,
                add_bos=True,
                add_eos=True,
                min_count=self.tgt_min_freq,
                lowercase=self.lowercase,
            )
            yin_before_vocab = Vocab(
                add_padding=True,
                add_bos=True,
                add_eos=True,
                min_count=self.tgt_min_freq,
                lowercase=self.lowercase,
            )

            yin_after_vocab = Vocab(
                add_padding=True,
                add_bos=True,
                add_eos=True,
                min_count=self.tgt_min_freq,
                lowercase=self.lowercase,
            )

            src_tag_vocab = Vocab(
                add_padding=True,
                add_bos=True,
                add_eos=True,
                min_count=1,
                lowercase=self.lowercase,
            )

            src_vocab = Vocab(
                add_padding=True,
                add_bos=True,
                add_eos=True,
                min_count=self.src_min_freq,
                lowercase=self.lowercase,
            )

            all_vocab = Vocab(
                add_padding=True,
                min_count=self.src_min_freq,
                lowercase=self.lowercase,
            )

            all_vocab.add_tokenized_sentence(["<BLANK>"])

            for example in tqdm(self.train, desc="Producing Vocabularies..."):
                limit = self.max_len if self.max_len else None
                tgt_tokens = list(example["tgt"])
                tgt_vocab.add_tokenized_sentence(tgt_tokens[:limit])
                all_vocab.add_tokenized_sentence(tgt_tokens[:limit])

                src_tokens = list(example["src"])
                src_vocab.add_tokenized_sentence(src_tokens[:limit])
                all_vocab.add_tokenized_sentence(src_tokens[:limit])

                yin_after = list(example["yin_after"])
                yin_after_vocab.add_tokenized_sentence(yin_after[:limit])

                yin_before = list(example["yin_before"])
                yin_before_vocab.add_tokenized_sentence(yin_before[:limit])

                src_tags = list(example["src_tag"])
                src_tag_vocab.add_tokenized_sentence(src_tags[:limit])

            tgt_vocab.finish()
            src_tag_vocab.finish()
            src_vocab.finish()
            yin_after_vocab.finish()
            yin_before_vocab.finish()
            all_vocab.finish()
            self.vocab = VocabTuple(
                src_vocab,
                src_tag_vocab,
                tgt_vocab,
                yin_before_vocab,
                yin_after_vocab,
                all_vocab,
            )
            with open(save_path, "wb") as f:
                pickle.dump(self.vocab, f)

        else:
            with open(save_path, "rb") as f:
                vocab = pickle.load(f)
                if not hasattr(vocab, "all"):
                    self.vocab = Vocab(
                        vocab.src_vocab,
                        vocab.src_tag_vocab,
                        vocab.tgt_vocab,
                        vocab.yin_before_vocab,
                        vocab.yin_after_vocab,
                        None,
                    )
                else:
                    self.vocab = vocab
