import json
import csv
import os
from itertools import chain
from difflib import SequenceMatcher
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

from wikigen.nlp import WordTokenizer
from wikigen.settings import EDITS_PATH, DATA_PATH

qt21_data_path = os.path.join(DATA_PATH, "QT21")

en_de_mqm_data_file_path = os.path.join(
    qt21_data_path, "QT21_mqm-data", "de-en.smt.csv"
)

en_de_pe_data_file_path = os.path.join(
    qt21_data_path, "QT21_pe-data", "de-en.smt"
)


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


def get_pe_types(string_data):
    xml_data = BeautifulSoup(string_data)
    output = []

    for item in xml_data.find_all("mqm:issue"):
        value = item.get("type")
        if value is not None:
            output.append(value)

    return set(output)


def get_pe_agent(string_data):
    xml_data = BeautifulSoup(string_data)
    output = []

    for item in xml_data.find_all("mqm:issue"):
        value = item.get("agent")
        if value is not None:
            output.append(value)

    return output


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


def parse_en_de_mqm(output_folder, tknzr):
    """
    Parse the MOM data in QT21 and choose
    the most common classes to use for multi-class 
    multi-label classification

    Classes selected:
        {'Addition',
        'Extraneous',
        'Incorrect',
        'Missing',
        'Mistranslation',
        'Omission',
        'Spelling',
        'Typography',
        'Untranslated',
        'Word order'}

    Result: 2.6961722488038276 classes/example

    For this dataset it is not really interesting to
    study the behavior of different annotators, since
    all pe sentences have the same agent 
    (Aron Woeste, also Kim Harris for the
     200 exampels with two annotators)
    """

    with open(en_de_mqm_data_file_path) as f:
        header = csv.reader(f).__next__()
        print(header)
        reader = csv.DictReader(f, fieldnames=header)
        en_de_mqm_data = [item for item in reader]

    for example in en_de_mqm_data:
        example_labels = get_pe_types(example["mqm"]) | get_pe_types(
            example["mqm2"]
        )
        example["labels"] = example_labels

    filtered_en_de_mqm_data = [
        example for example in en_de_mqm_data if len(example["labels"]) > 0
    ]

    counter = Counter()
    for example in filtered_en_de_mqm_data:
        counter.update(example["labels"])

    classes = set(
        [label for label, frequency in counter.items() if frequency > 100]
    )

    usable_en_de_mqm_data = []
    for example in filtered_en_de_mqm_data:
        usable_example_labels = classes & example["labels"]
        if len(usable_example_labels) >= 1:
            usable_example = {}
            usable_example.update(example)
            usable_example["labels"] = list(usable_example_labels)
            usable_en_de_mqm_data.append(usable_example)

    output_file_path = os.path.join(output_folder, "qt21_en_de_mqm.jsonl")

    with open(output_file_path, "w") as f:
        for example in usable_en_de_mqm_data:

            # the MT generated translation
            before = example["target"]
            # the post-edited translation
            after = example["pe-output"]

            tknzd_orig = tknzr.tokenize(before)
            tknzd_mod = tknzr.tokenize(after)

            if tknzd_orig == tknzd_mod:
                continue

            yin_before, yin_after, yin_tags = build_yin_edit_sequence(
                tknzd_orig, tknzd_mod
            )

            json_data = {
                "id": example["mid"],
                "src_tag": yin_tags,
                "yin_after": yin_after,
                "yin_before": yin_before,
                "src": tknzd_orig,
                "tgt": tknzd_mod,
                "tgt_classes": {
                    label: 1 if label in example["labels"] else 0
                    for label in classes
                },
            }

            json_data = parse_ds(json_data)

            json_str = json.dumps(json_data)

            f.write(f"{json_str}\n")


def parse_en_de_pe(output_folder, tknzr):
    """
    Preprocesing involves simply selecting all the examples
    from this dataset, and adding the author
    """

    with open(en_de_pe_data_file_path) as f:
        header = [
            "sentence-id",
            "source",
            "mt",
            "pe-output",
            "target",
            "pe-score",
            "pe-time",
            "letter-keys",
            "digit-keys",
            "white-keys",
            "symbol-keys",
            "navigation-keys",
            "erase-keys",
            "copy-keys",
            "cut-keys",
            "paste-keys",
            "do-keys",
            "translator-id",
        ]

        reader = csv.DictReader(f, fieldnames=header, delimiter="\t")

        en_de_pe_data = [item for item in reader]

    output_file_path = os.path.join(output_folder, "qt21_en_de_pe.jsonl")

    with open(output_file_path, "w") as f:
        for example in en_de_pe_data:

            # the MT generated translation
            before = example["mt"]
            # the post-edited translation
            after = example["pe-output"]

            tknzd_orig = tknzr.tokenize(before)
            tknzd_mod = tknzr.tokenize(after)

            if tknzd_orig == tknzd_mod:
                continue

            yin_before, yin_after, yin_tags = build_yin_edit_sequence(
                tknzd_orig, tknzd_mod
            )

            json_data = {
                "id": example["sentence-id"],
                "src_tag": yin_tags,
                "yin_after": yin_after,
                "yin_before": yin_before,
                "src": tknzd_orig,
                "tgt": tknzd_mod,
                "author": example["translator-id"],
            }

            json_data = parse_ds(json_data)

            json_str = json.dumps(json_data)

            f.write(f"{json_str}\n")


if __name__ == "__main__":

    tknzr = WordTokenizer()

    parse_en_de_mqm(EDITS_PATH, tknzr)
    parse_en_de_pe(EDITS_PATH, tknzr)

