import json
import os
from itertools import chain
from difflib import SequenceMatcher
from collections import Counter

from wikigen.settings import EDITS_PATH, DATA_PATH


lang8_corpus_path = os.path.join(DATA_PATH, "lang8.bea19")
lang8_file_path = os.path.join(lang8_corpus_path, "lang8.train.auto.bea19.m2")


wi_and_locness_corpus_path = os.path.join(DATA_PATH, "wi+locness", "m2")


def parse_ds(d):
    indices = [i for i, x in enumerate(d["src_tag"]) if x != "equal"]
    changed = []
    for i in indices:
        word = (
            d["yin_before"][i]
            if d["yin_before"][i] != "__EMPTY__"
            else d["yin_after"][i]
        )
        changed.append(word)
    d["changed"] = changed
    return d


def m2_to_sents(file_path, annotator_id=0):
    # Apply the edits of a single annotator to generate the corrected sentences.
    m2 = open(file_path).read().strip().split("\n\n")
    out = []
    # Do not apply edits with these error types
    skip = {"noop", "UNK", "Um"}

    for sent in m2:
        sent = sent.split("\n")
        cor_sent = sent[0].split()[1:]  # Ignore "S "
        original_sentence = cor_sent.copy()
        edits = sent[1:]
        offset = 0
        for edit in edits:
            edit = edit.split("|||")
            if edit[1] in skip:
                continue  # Ignore certain edits
            coder = int(edit[-1])
            if coder != annotator_id:
                continue  # Ignore other coders
            span = edit[0].split()[1:]  # Ignore "A "
            start = int(span[0])
            end = int(span[1])
            cor = edit[2].split()
            cor_sent[start + offset : end + offset] = cor
            offset = offset - (end - start) + len(cor)
        if original_sentence != cor_sent:
            out.append((original_sentence, cor_sent))
    return out


def build_yin_edit_sequence(tokens_before, tokens_after):

    # Return list of 5-tuples describing how to turn `tokens_before` into `tokens_after`.
    s = SequenceMatcher(None, tokens_before, tokens_after)

    meta_sequence_before = []
    meta_sequence_after = []
    tags = []

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            # tokens_before[i1:i2] is equal to tokens_after[j1:j2]
            for token in tokens_before[i1:i2]:
                meta_sequence_before.append(token)
                meta_sequence_after.append(token)
                tags.append("equal")

        if tag == "replace":
            i_diff = i2 - i1
            j_diff = j2 - j1
            i_j_diff = i_diff - j_diff
            for token in tokens_before[i1:i2]:
                meta_sequence_before.append(token)

            for token in tokens_after[j1:j2]:
                meta_sequence_after.append(token)

            for i in range(max(i_diff, j_diff)):
                tags.append("replace")

            if i_j_diff < 0:
                for i in range(abs(i_j_diff)):
                    meta_sequence_before.append("__EMPTY__")
            if i_j_diff > 0:
                for i in range(i_j_diff):
                    meta_sequence_after.append("__EMPTY__")

        if tag == "insert":
            #  i1 == i2 in this case
            for token in tokens_after[j1:j2]:
                meta_sequence_before.append("__EMPTY__")
                meta_sequence_after.append(token)
                tags.append("insert")

        if tag == "delete":
            #  j1 == j2 in this case
            for token in tokens_before[i1:i2]:
                meta_sequence_before.append(token)
                meta_sequence_after.append("__EMPTY__")
                tags.append("delete")

    assert len(meta_sequence_before) == len(meta_sequence_after)
    assert len(meta_sequence_before) == len(tags)

    return meta_sequence_before, meta_sequence_after, tags


def parse_lang8(output_folder):

    output = m2_to_sents(lang8_file_path)

    output_file_path = os.path.join(output_folder, "lang8.jsonl")

    with open(output_file_path, "w") as f:
        for i, example in enumerate(output):

            before, after = example

            tknzd_orig = before
            tknzd_mod = after

            if tknzd_orig == tknzd_mod:
                continue

            yin_before, yin_after, yin_tags = build_yin_edit_sequence(
                tknzd_orig, tknzd_mod
            )

            json_data = {
                "id": i,
                "src_tag": yin_tags,
                "yin_after": yin_after,
                "yin_before": yin_before,
                "src": tknzd_orig,
                "tgt": tknzd_mod,
            }

            json_data = parse_ds(json_data)

            json_str = json.dumps(json_data)

            f.write(f"{json_str}\n")


def parse_wi_plus_locness(output_folder):
    """
    Preprocesing involves simply selecting all the examples
    from this dataset, and adding the author
    """

    # A.train.gold.bea19.m2
    # B.train.gold.bea19.m2
    # C.train.gold.bea19.m2

    # A.dev.gold.bea19.m2
    # B.dev.gold.bea19.m2
    # C.dev.gold.bea19.m2
    # N.dev.gold.bea19.m2

    labels = ["A", "B", "C", "N"]

    examples = []

    for label in labels:
        train_file_name = f"{label}.train.gold.bea19.m2"
        dev_file_name = f"{label}.dev.gold.bea19.m2"

        train_file_path = os.path.join(
            wi_and_locness_corpus_path, train_file_name
        )

        dev_file_path = os.path.join(wi_and_locness_corpus_path, dev_file_name)

        try:
            train_output = m2_to_sents(train_file_path)
        except IOError:
            train_output = []

        try:
            dev_output = m2_to_sents(dev_file_path)
        except IOError:
            dev_output = []

        for src, tgt in train_output:
            example = {
                "src": src,
                "tgt": tgt,
                "cefr_level": label,
                "split": "train",
            }
            examples.append(example)

        for src, tgt in dev_output:
            example = {
                "src": src,
                "tgt": tgt,
                "cefr_level": label,
                "split": "dev",
            }
            examples.append(example)

    output_file_path = os.path.join(output_folder, "wi_plus_locness.jsonl")

    with open(output_file_path, "w") as f:
        for i, example in enumerate(examples):

            # the MT generated translation
            before = example["src"]
            # the post-edited translation
            after = example["tgt"]

            tknzd_orig = before
            tknzd_mod = after

            if tknzd_orig == tknzd_mod:
                continue

            yin_before, yin_after, yin_tags = build_yin_edit_sequence(
                tknzd_orig, tknzd_mod
            )

            json_data = {
                "id": i,
                "src_tag": yin_tags,
                "yin_after": yin_after,
                "yin_before": yin_before,
                "src": tknzd_orig,
                "tgt": tknzd_mod,
                "tgt_class": {
                    label: 1 if label == example["cefr_level"] else 0
                    for label in labels
                },
                "split": example["split"],
            }

            json_data = parse_ds(json_data)

            json_str = json.dumps(json_data)

            f.write(f"{json_str}\n")


if __name__ == "__main__":

    parse_lang8(EDITS_PATH)
    parse_wi_plus_locness(EDITS_PATH)
