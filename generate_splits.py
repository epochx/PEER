#!/usr/bin/env python
# -*-coding: utf8 -*-

import argparse
import os
import json

from wikigen.data import split_list
from wikigen.settings import EDIT_AUTOENCODER_PARSED_EDITS, DATASET_NAMES, SPLITS_PATH

desc = "Help for preprocess"

parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--dataset",
                    help="Project name",
                    choices=DATASET_NAMES)

parser.add_argument("--splits", '-s',
                    type=str,
                    default='0.7/0.1/0.2',
                    help='Splitting ratios. Default: 0.7/0.1/0.2')

args = parser.parse_args()

train, valid, test = map(float, args.splits.split('/'))
if train + valid + test != 1.0:
    print('Splits do not add to 1')
    exit()

parsed_commits_file_name = args.dataset

examples = []
with open(os.path.join(
        EDIT_AUTOENCODER_PARSED_EDITS, f'{args.dataset}.jsonl')) as f:
    for line in f.readlines():
        example = json.loads(line.strip())
        examples.append(example)

len_parsed_commits = len(examples)
print(f'Read {len_parsed_commits} lines')


if any(['split' in example for example in examples]):

    print('Splitting based on data splits (provided ratios ignored)')

    train_indices = [
        i for i in range(len_parsed_commits)
        if examples[i]['split'] == 'train']

    valid_indices = [
        i for i in range(len_parsed_commits)
        if examples[i]['split'] in ('valid', 'dev')]
    
    test_indices = [
        i for i in range(len_parsed_commits)
        if examples[i]['split'] == 'test']

    if not test_indices:
        test_indices = valid_indices 

else:

    print('Splitting based on provided split sizes')

    if not os.path.exists(SPLITS_PATH):
        os.makedirs(SPLITS_PATH)

    parsed_commits_indices = list(range(len_parsed_commits))

    train_indices, valid_indices, test_indices = split_list(
        parsed_commits_indices, train, valid, test)

base_indices_file_path = os.path.join(SPLITS_PATH, parsed_commits_file_name)

train_indices_file_path = base_indices_file_path + ".train.txt"
with open(train_indices_file_path, "w") as f:
    f.write('\n'.join(map(str, train_indices)))

valid_indices_file_path = base_indices_file_path + ".valid.txt"
with open(valid_indices_file_path, "w") as f:
    f.writelines('\n'.join(map(str, valid_indices)))

test_indices_file_path = base_indices_file_path + ".test.txt"
with open(test_indices_file_path, "w") as f:
    f.writelines('\n'.join(map(str, test_indices)))

print('Generated splits')

