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
from wikigen.eval.bleu import bleu
from wikigen.eval.gleu import gleu

from wikigen.data import AutoEncoderDataset
from wikigen.logger import Logger


def calculate_vae_loss(batch_size, recon_loss, kl_loss, step, anneal_function):

    recon = recon_loss["recon"]
    bow_loss = recon_loss["bow"]

    if anneal_function is None:
        return (recon + bow_loss) / batch_size, 0, 0

    KL_weight = anneal_function(step)
    if step % 40 == 0:
        tqdm.tqdm.write(
            "KL Weight: " + colors.colorize(f"{KL_weight:.7f}", "blue")
        )

    weighted_kl = KL_weight * kl_loss
    weighted_class = min(KL_weight * 2, 1)
    if kl_loss == 0:
        return (recon + bow_loss) / batch_size, 0, 0
    else:
        return (
            (recon + weighted_kl + bow_loss) / batch_size,
            weighted_kl.item(),
            KL_weight,
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate on test"
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
        "--beam_size", type=int, default=5, help="beam size (default: 32)"
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Which split to encode. Default: test",
    )

    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"], help="Device"
    )

    parser.add_argument(
        "--best",
        action="store_true",
        help="To load the best model, otherwise the last epoch available.",
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

    def generate_output(
        input_ids,
        src_sequences,
        tgt_sequences,
        gen_sequences,
        tgt_probs=None,
        gen_probs=None,
        src_tag_sequences=None,
        p_gens=None,
        edit_representations=None,
    ):

        length = len(input_ids)
        output = []
        for i in range(length):

            output_i = {
                "id": input_ids[i],
                "src": src_sequences[i],
                "tgt": tgt_sequences[i],
                "gen": gen_sequences[i],
            }

            if gen_probs:
                output_i["gen_probs"] = gen_probs[i]

            if tgt_probs:
                output_i["tgt_probs"] = tgt_probs[i]

            if src_tag_sequences:
                output_i["src_tags"] = src_tag_sequences[i]

            if edit_representations:
                output_i["edit_repr"] = edit_representations[i]
            output.append(output_i)

        return output

    # output_save_path = os.path.join(run_args.model, output_file_name)

    batches = BatchIterator(
        vocab, data, run_args.batch_size, YinEditBatch, max_len=args.max_len,
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
    
    batch_size = run_args.batch_size

    batch_i = 0
    total_valid_loss = 0
    total_valid_class_loss = 0
    total_valid_nllloss = 0
    total_valid_klloss = 0
    total_valid_ppl = 0
    valid_counter = 0
    gen_sequences = []
    src_sequences = []
    src_tag_sequences = []
    tgt_sequences = []
    input_ids = []
    attentions = []

    for batch in tqdm.tqdm(batches, desc=f"Testing on {run_args.split}... "):
        valid_counter += 1
        model.eval()
        src_batch_sequences = batch.src_batch.sequences
        src_batch_tag_sequences = batch.src_batch.tag_sequences
        tgt_output_batch_sequences = batch.tgt_input_batch.sequences[:, 1:]

        try:
            batch.to_torch(device=device)

            model_out, _ = model.validation(
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
                vocab=dataset.vocab,
                beam_size=run_args.beam_size,
                max_len=args.max_len,
                changed=batch.changed.sequences,
                changed_mask=batch.changed.masks,
                added_edit_batch_tuple=batch.added_batch,
                removed_edit_batch_tuple=batch.removed_batch,
            )

            loss, weighted_kl, kl_weight = calculate_vae_loss(
                src_batch_sequences.shape[0],
                model_out["loss"],
                model_out["KLD"],
                valid_counter,
                lambda x: 1,
            )

            total_valid_loss += loss.item()
            if args.num_classes:
                total_valid_class_loss = (
                    model_out["loss"]["class"].item() / batch_size
                )
            else:
                total_valid_class_loss = 0
            total_valid_nllloss += (
                model_out["loss"]["recon"].item() / batch_size
            )
            total_valid_klloss += model_out["KLD_item"] / batch_size
            total_valid_ppl += model_out["PPL"].item()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if "out of memory" in str(e):
                print(
                    "| WARNING: ran out of memory, skipping batch. "
                    "if this happens frequently, decrease test_batch_size or "
                    "truncate the inputs to the model."
                )
                continue
            else:
                raise e

        for id_sequence in model_out["preds"]:

            gen_sequence_i = dataset.vocab.tgt.indices2tokens(
                id_sequence, ignore_ids=tgt_ignore_ids,
            )
            gen_sequences.append(" ".join(gen_sequence_i))

        for i, id_sequence in enumerate(src_batch_sequences):
            src_sequence_i = dataset.vocab.src.indices2tokens(
                id_sequence, ignore_ids=src_ignore_ids,
            )
            src_sequences.append(" ".join(src_sequence_i))

        if src_batch_tag_sequences is not None:
            for i, id_sequence in enumerate(src_batch_tag_sequences):
                src_tag_sequence_i = dataset.vocab.src_tag.indices2tokens(
                    id_sequence, ignore_ids=tag_ignore_ids
                )

                src_tag_sequences.append(" ".join(src_tag_sequence_i))

        for i, id_sequence in enumerate(tgt_output_batch_sequences):
            tgt_sequence_i = dataset.vocab.tgt.indices2tokens(
                id_sequence, ignore_ids=tgt_ignore_ids,
            )
            tgt_sequences.append(" ".join(tgt_sequence_i))

        input_ids += batch.ids_batch
        batch_i += 1

    total_valid_loss = 1.0 * total_valid_loss / batch_i
    total_valid_nllloss = 1.0 * total_valid_nllloss / batch_i
    total_valid_class_loss = 1.0 * total_valid_class_loss / batch_i
    total_valid_ppl = 1.0 * total_valid_ppl / batch_i
    total_valid_klloss = 1.0 * total_valid_klloss / batch_i
    if run_args.device == "cuda":
        torch.cuda.empty_cache()

    output = generate_output(
        input_ids,
        src_sequences,
        tgt_sequences,
        gen_sequences,
        src_tag_sequences=src_tag_sequences,
    )

    gold_sequences = {item["id"]: item["tgt"] for item in output}
    pred_sequences = {item["id"]: item["gen"] for item in output}
    src_sequences = {item["id"]: item["src"] for item in output}

    # valid_meteor = meteor(gold_sequences, pred_sequences)
    valid_meteor = 0
    valid_bleu = bleu(gold_sequences, pred_sequences)
    # valid_meant = meant(gold_sequences, pred_sequences)["fscore"]
    valid_meant = 0

    # valid_errant = errant_metric(
    #     src_sequences, gold_sequences, pred_sequences)[-1]
    valid_errant = 0
    valid_gleu = gleu(gold_sequences, src_sequences, pred_sequences)

    try:
        current_log = {
            f"{run_args.split}/Loss": total_valid_loss,
            f"{run_args.split}/NLLLoss": total_valid_nllloss,
            f"{run_args.split}/KLLoss (UW)": total_valid_klloss,
            f"{run_args.split}/Class Loss (UW)": total_valid_class_loss,
            f"{run_args.split}/BLEU": valid_bleu,
            f"{run_args.split}/ERRANT": valid_errant,
            f"{run_args.split}/GLEU": valid_gleu,
            f"{run_args.split}/METEOR": valid_meteor,
            f"{run_args.split}/MEANT": valid_meant,
            f"{run_args.split}/PPL": total_valid_ppl,
        }
    except AttributeError:
        current_log = {
            f"{run_args.split}/Loss": total_valid_loss,
            f"{run_args.split}/NLLLoss": total_valid_nllloss,
            f"{run_args.split}/KLLoss (UW)": total_valid_klloss,
            f"{run_args.split}/Class Loss (UW)": total_valid_class_loss,
            f"{run_args.split}/BLEU": valid_bleu,
            f"{run_args.split}/ERRANT": valid_errant,
            f"{run_args.split}/GLEU": valid_gleu,
            f"{run_args.split}/METEOR": valid_meteor,
            f"{run_args.split}/MEANT": valid_meant,
            f"{run_args.split}/PPL": total_valid_ppl,
        }

    for key, value in current_log.items():
        print(f"{key}: {value:14.3f}")
