#!/usr/bin/env python
# -*-coding: utf8 -*-

import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))

HOME = os.environ["HOME"]

DATA_PATH = os.path.join(HOME, "data", "PEER")
SPLITS_PATH = os.path.join(DATA_PATH, "splits")
EDITS_PATH = os.path.join(DATA_PATH, "edits")

RESULTS_PATH = os.path.join(HOME, "results", "PEER")

_DB_NAME = "runs.db"

PARAM_IGNORE_LIST = [
    "results_path",
    "overwrite",
    "force_dataset_reload",
    "verbose",
    "write_mode",
]

DATABASE_CONNECTION_STRING = "sqlite:///" + os.path.join(RESULTS_PATH, _DB_NAME)

try:
    DATASET_NAMES = [
        name.replace(".jsonl", "") for name in os.listdir(EDITS_PATH)
    ]
except FileNotFoundError:
    DATASET_NAMES = []

