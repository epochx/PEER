import json
import numpy as np
from difflib import SequenceMatcher
import pandas as pd
from itertools import chain
from wikigen.nlp import WordTokenizer
import os
from wikigen.settings import EDITS_PATH
from tqdm import tqdm


wae_data_path = os.path.join(DATA_PATH, "wiki_atomic_edits")


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


tknzr = WordTokenizer()
with open(EDITS_PATH + "/insertions_deletions.jsonl", "w") as out_file:
    counter_list = np.arange(10000)
    np.random.shuffle(counter_list)
    print("Indices Shuffled")
    counter = 0
    for tsv in ["deletions.tsv", "insertions.tsv"]:

        current_file = pd.read_csv(os.path.join(wae_path, tsv), delimiter="\t")

        for i in tqdm(range(int(10000 / 2))):
            tknzd_mod = tknzr.tokenize(
                current_file["edited_sentence"][i + 6000000]
            )
            tknzd_orig = tknzr.tokenize(
                current_file["base_sentence"][i + 6000000]
            )

            yin_before, yin_after, yin_tags = build_yin_edit_sequence(
                tknzd_orig, tknzd_mod
            )

            json_data = {
                "id": int(counter_list[counter]),
                "src_tag": yin_tags,
                "yin_after": yin_after,
                "yin_before": yin_before,
                "src": tknzd_orig,
                "tgt": tknzd_mod,
            }

            json_str = json.dumps(json_data)

            out_file.write(f"{json_str}\n")

            counter += 1
