import argparse
from itertools import chain
import json
import os

import torch
import tqdm

from wikigen.config import AttrDict
from wikigen.settings import RESULTS_PATH, DATASET_NAMES
import wikigen.colors as colors
from wikigen.model.edit_vae import EditVAE
from wikigen.model.batch import BatchIterator, YinEditBatch

from wikigen.data import AutoEncoderDataset
from wikigen.logger import Logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sample from latent space and store resulting embeddings"
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the trained model folder to load",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size (default: 32)"
    )

    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test", "all"],
        help="Which split to encode. Default: valid",
    )

    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"], help="Device"
    )

    parser.add_argument(
        "--best",
        action="store_true",
        help="To load the best model, otherwise the last epoch available.",
    )

    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        help="To encode a different dataset that the one the model was trained on.",
    )

    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="To save in JSONL format.",
    )

    run_args = parser.parse_args()

    model_save_path = os.path.join(run_args.model, "model.pth")
    hyperparams_save_path = os.path.join(run_args.model, "hyperparams.json")

    with open(hyperparams_save_path) as f:
        args = AttrDict(json.load(f))

    device = torch.device(run_args.device)
    if run_args.device == "cpu":
        torch.manual_seed(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)

    dataset = AutoEncoderDataset(
        RESULTS_PATH,
        args.dataset,
        args.min_freq,
        joint=True,
        max_len=args.max_len,
        force_reload=False,
        lowercase=args.lowercase,
        generate=True,
    )

    vocab = dataset.vocab

    if run_args.best:
        model = torch.load(model_save_path + ".best").to(device=device)
    else:
        model = torch.load(model_save_path).to(device=device)

    if run_args.dataset:
        dataset = AutoEncoderDataset(
            RESULTS_PATH,
            run_args.dataset,
            1,
            joint=True,
            force_reload=True,
            lowercase=args.lowercase,
            generate=True,
        )

    if run_args.split == "train":
        data = dataset.train
    elif run_args.split == "valid":
        data = dataset.valid
    elif run_args.split == "test":
        data = dataset.test
    else:
        data = list(
            chain(
                dataset.train,
                dataset.valid,
                dataset.test
            )
        )

    if run_args.dataset:
        output_file_name = f"embeddings_{run_args.dataset}_{run_args.split}.json"
    else:
        output_file_name = f"embeddings_{run_args.split}.json"
    
    if run_args.jsonl:
        output_file_name += "l"


    output_save_path = os.path.join(run_args.model, output_file_name)

    if os.path.exists(output_save_path):
        print(f"Embeddings for {output_file_name} already exists, skipping.")
        exit()

    batches = BatchIterator(
        vocab, data, args.batch_size, YinEditBatch, max_len=args.max_len,
    )

    src_ignore_ids = set(
        [
            dataset.vocab.src.PAD.hash,
            dataset.vocab.src.BOS.hash,
            dataset.vocab.src.EOS.hash,
        ]
    )
    tgt_ignore_ids = set(
        [
            dataset.vocab.tgt.PAD.hash,
            dataset.vocab.tgt.BOS.hash,
            dataset.vocab.tgt.EOS.hash,
        ]
    )
    tag_ignore_ids = set(
        [
            dataset.vocab.src_tag.PAD.hash,
            dataset.vocab.src_tag.BOS.hash,
            dataset.vocab.src_tag.EOS.hash,
        ]
    )

    save_output = []

    for batch in tqdm.tqdm(batches, desc="Encoding... "):

        batch.to_torch(device=device)

        batch_mu = model.encode(
            batch.yin_before.sequences,
            batch.yin_after.sequences,
            batch.src_batch.tag_sequences,
            batch.src_batch.sequences,
            batch.tgt_input_batch.sequences,
            seq_lens={
                "edit": batch.yin_before.lengths,
                "before": batch.src_batch.lengths,
                "after": batch.tgt_input_batch.lengths,
            },
            src_batch_mask=batch.src_batch.masks,
            tgt_batch_mask=batch.tgt_input_batch.masks,
            added_edit_batch_tuple=batch.added_batch,
            removed_edit_batch_tuple=batch.removed_batch,
        )

        batch_latent = batch_mu.detach()

        if run_args.device == "cuda":
            batch_latent = batch_latent.to(torch.device("cpu")).numpy()
        else:
            batch_latent = batch_latent.numpy()

        for example, batch_latent_i in zip(batch.examples, batch_latent):
            datum = example.copy()
            list_latent_i = batch_latent_i.tolist()
            datum["edit_representation"] = list_latent_i
            save_output.append(datum)


    print(f"Saving to {output_save_path}")

    if run_args.jsonl:
        with open(output_save_path, "w") as f:
            for output in save_output:
                json_output = json.dumps(output)
                f.write(f"{output}\n")
    else:    
        with open(output_save_path, "w") as f:
            json.dump(save_output, f)
