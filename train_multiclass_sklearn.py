#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring

from wikigen.settings import SPLITS_PATH, DATASET_NAMES

np.random.seed(2)
torch.cuda.manual_seed(2)

parser = argparse.ArgumentParser(description="Sample from a trained RVAE")

parser.add_argument("json_file_path", type=str, help="Path to data")

parser.add_argument(
    "--device", type=str, help="Device", choices=["gpu", "cpu"], default="gpu"
)

parser.add_argument(
    "--num_layers", type=int, help="Device", choices=[1, 2], default=1,
)


class TorchNeuralNetClassifier(nn.Module):
    def __init__(self, num_units=100, nonlin=nn.ReLU(), num_layers=1):
        super(TorchNeuralNetClassifier, self).__init__()
        if num_layers not in [1, 2]:
            raise NotImplementedError
        self.dense0 = nn.Linear(input_size, num_units)
        self.nonlin = nonlin
        if num_layers == 2:
            self.dense1 = nn.Linear(num_units, 200)
            self.output = nn.Linear(200, output_size)
        else:
            self.dense1 = None
            self.output = nn.Linear(num_units, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        if self.dense1 is not None:
            X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


if __name__ == "__main__":

    args = parser.parse_args()

    multi_label_binarizer = MultiLabelBinarizer()

    file_name = os.path.basename(args.json_file_path)

    matches = [dataset_name in file_name for dataset_name in DATASET_NAMES]
    assert any(matches)
    assert sum(matches) == 1

    name = DATASET_NAMES[matches.index(True)]

    with open(args.json_file_path) as f:
        examples = json.load(f)

    base_indices_file_path = os.path.join(SPLITS_PATH, name)

    train_indices_file_path = base_indices_file_path + ".train.txt"
    with open(train_indices_file_path) as f:
        train_indices = np.array(
            list(map(lambda x: int(x.strip()), f.readlines()))
        )

    valid_indices_file_path = base_indices_file_path + ".valid.txt"
    with open(valid_indices_file_path) as f:
        valid_indices = np.array(
            list(map(lambda x: int(x.strip()), f.readlines()))
        )

    test_indices_file_path = base_indices_file_path + ".test.txt"
    with open(test_indices_file_path) as f:
        test_indices = np.array(
            list(map(lambda x: int(x.strip()), f.readlines()))
        )

    X = np.asarray([example["edit_representation"] for example in examples])

    if "tgt_class" in examples[0]:

        labels = list(examples[0]["tgt_class"].keys())
        label2index = {label: index for index, label in enumerate(labels)}

        Y = [
            [
                label2index[label]
                for label, value in example["tgt_class"].items()
                if value == 1
            ][0]
            for example in examples
        ]

    else:
        Y = [example["category"] for example in examples]

    Y = np.asarray(Y)

    if args.device == "cpu":
        neural_net = MLPClassifier(max_iter=3000, random_state=2,)
        if args.num_layers == 1:
            param_grid = {"hidden_layer_sizes": [[100,], [200,], [300,]]}
        else:
            param_grid = {
                "hidden_layer_sizes": [[100, 200,], [200, 200,], [300, 200,]]
            }

    else:
        X = torch.from_numpy(X).float()
        Y = torch.tensor(Y).squeeze().long()

        input_size = X.size(1)
        output_size = int(Y.max() + 1)

        def scoring_function(net, X=None, y=None):
            return net.history[-1]["train_loss"]

        neural_net = NeuralNetClassifier(
            TorchNeuralNetClassifier,
            max_epochs=3000,
            module__num_layers=args.num_layers,
            iterator_train__shuffle=True,
            train_split=False,
            device=torch.device("cuda"),
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=0.0001,
            optimizer__lr=0.001,
            batch_size=min(200, X.shape[0]),
            callbacks=[
                (
                    "t_loss",
                    EpochScoring(
                        scoring_function, on_train=True, name="epoch_train_loss"
                    ),
                ),
                (
                    "stop",
                    EarlyStopping(monitor="epoch_train_loss", patience=10,),
                ),
            ],
        )

        param_grid = {
            "module__num_units": [100, 200, 300],
        }

    model = GridSearchCV(
        neural_net,
        param_grid,
        cv=[(train_indices, valid_indices)],
        verbose=1,
        n_jobs=3,
    )

    model.fit(X, Y)

    print(f"Train Acc: {model.score(X[train_indices], Y[train_indices])}")
    print(f"Valid Acc: {model.score(X[valid_indices], Y[valid_indices])}")
    print(f"Test Acc: {model.score(X[test_indices], Y[test_indices])}")

