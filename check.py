import argparse
import os
import json
import numpy as np


from wikigen.settings import EDITS_PATH, DATASET_NAMES, SPLITS_PATH

only_insert = set(["equal", "insert"])
only_delete = set(["equal", "delete"])
only_replace = set(["equal", "replace"])

desc = "Help for check"

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--dataset", help="Project name", choices=DATASET_NAMES)


if __name__ == "__main__":

    args = parser.parse_args()

    parsed_commits_file_name = args.dataset

    examples = []
    with open(os.path.join(EDITS_PATH, f"{args.dataset}.jsonl")) as f:
        for line in f.readlines():
            example = json.loads(line.strip())
            examples.append(example)

    src_lens = []
    tgt_lens = []

    only_insert_list = []
    only_delete_list = []
    only_replace_list = []

    count = 0
    for example in examples:
        if example["src"] == example["tgt"]:
            count += 1
            print(count)

        src_lens.append(len(example["src"]))
        tgt_lens.append(len(example["tgt"]))

        if set(example["src_tag"]) == only_insert:
            only_insert_list.append(example)

        if set(example["src_tag"]) == only_delete:
            only_delete_list.append(example)

        if set(example["src_tag"]) == only_replace:
            only_replace_list.append(example)

    print(f"Only insertions: {len(only_insert_list)/len(examples)}")
    print(f"Only deletions: {len(only_delete_list)/len(examples)}")
    print(f"Only replacements: {len(only_replace_list)/len(examples)}")

    print(f"SRC mean len: {np.mean(src_lens)}, SRC std len: {np.std(src_lens)}")
    print(f"TGT mean len: {np.mean(tgt_lens)}, TGT std len: {np.std(tgt_lens)}")

