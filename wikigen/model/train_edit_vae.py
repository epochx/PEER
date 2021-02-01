import json
import os
import subprocess

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import tqdm

import wikigen.colors as colors
from wikigen.model.edit_vae import EditVAE
from wikigen.model.batch import BatchIterator, YinEditBatch

from wikigen.model.scheduler import Scheduler
from wikigen.model.optimizers import optimizers
from wikigen.data import AutoEncoderDataset
from wikigen.logger import Logger
from wikigen.eval.bleu import bleu
from wikigen.eval.gleu import gleu

# FIXME: reach_iter=25000
def get_logistic_anneal_function(k, x0, reach_iter=25000):
    def anneal_function(step):
        scaled_iter = step / reach_iter
        return float(1 / (1 + (np.exp(-13 * (scaled_iter - 0.5)))))

    return anneal_function


def get_linear_anneal_function(x0):
    def anneal_function(step):
        return min(1, step / x0)

    return anneal_function


def get_frange_cycle_linear(
    start=0.0, stop=1.0, n_epoch=80000, n_cycle=6, ratio=0.25
):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1

    def anneal_function(step):
        return L[step]

    return anneal_function


def get_kl_annealing_function(name, **kwargs):

    if name == "logistic":
        anneal_function = get_logistic_anneal_function(
            kwargs["k"], kwargs["x0"]
        )

    elif name == "linear":
        anneal_function = get_linear_anneal_function(kwargs["x0"])

    elif name == "cyclical":
        anneal_function = get_frange_cycle_linear()

    else:
        raise NotImplementedError
    return anneal_function


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


def main(args):

    device = torch.device(args.device)
    if args.device == "cpu":
        torch.manual_seed(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)

    dataset = AutoEncoderDataset(
        args.results_path,
        args.dataset,
        args.min_freq,
        joint=True,
        max_len=args.max_len,
        force_reload=args.force_dataset_reload,
        lowercase=args.lowercase,
        generate=True,
    )

    logger = Logger(
        args,
        model_name="EditVAE",
        write_mode=args.write_mode,
        hash_value=None if args.hash is None else args.hash,
    )

    model_id = logger.hash
    results_path = logger.run_savepath

    embeddings = nn.Embedding(
        len(dataset.vocab.src),
        args.encoder_embedding_size,
        padding_idx=dataset.vocab.src.PAD.hash,
    )
    embeddings.unk_idx = dataset.vocab.src.UNK.hash

    decoder_embeddings = nn.Embedding(
        len(dataset.vocab.tgt),
        args.encoder_embedding_size,
        padding_idx=dataset.vocab.tgt.PAD.hash,
    )
    decoder_embeddings.unk_idx = dataset.vocab.tgt.UNK.hash

    if args.edit_encoder == "yin":

        input_embeddings = nn.Embedding(
            len(dataset.vocab.yin_before),
            args.encoder_embedding_size,
            padding_idx=dataset.vocab.yin_before.PAD.hash,
        )
        input_embeddings.unk_idx = dataset.vocab.yin_before.UNK.hash

        output_embeddings = nn.Embedding(
            len(dataset.vocab.yin_after),
            args.encoder_embedding_size,
            padding_idx=dataset.vocab.yin_after.PAD.hash,
        )
        output_embeddings.unk_idx = dataset.vocab.yin_after.UNK.hash

        encoder_tag_embeddings = nn.Embedding(
            len(dataset.vocab.src_tag),
            args.encoder_tag_embedding_size,
            padding_idx=dataset.vocab.src_tag.PAD.hash,
        )
    else:

        input_embeddings = nn.Embedding(
            len(dataset.vocab.all),
            args.encoder_embedding_size,
            padding_idx=dataset.vocab.all.PAD.hash,
        )
        output_embeddings = None
        encoder_tag_embeddings = None

    train_batches = BatchIterator(
        dataset.vocab,
        dataset.train,
        args.batch_size,
        YinEditBatch,
        max_len=args.max_len,
    )

    valid_batches = BatchIterator(
        dataset.vocab,
        dataset.valid,
        args.test_batch_size,
        YinEditBatch,
        max_len=args.max_len,
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
    test_batches = BatchIterator(
        dataset.vocab, dataset.test, 2, YinEditBatch, max_len=args.max_len
    )

    # MODEL
    model = EditVAE(
        embeddings={
            "before": embeddings,
            "after": decoder_embeddings,
            "edit_before": input_embeddings,
            "edit_after": output_embeddings,
            "tags": encoder_tag_embeddings,
        },
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        dropout={
            "edit": {
                "input": args.edit_input_dropout,
                "output": args.edit_output_dropout,
            },
            "before": {
                "input": args.before_input_dropout,
                "output": args.before_output_dropout,
                "word": args.before_word_dropout,
            },
            "after": {
                "input": args.after_input_dropout,
                "output": args.after_output_dropout,
                "word": args.after_word_dropout,
            },
        },
        latent_size=args.latent_size,
        use_kl=args.use_kl,
        bow_loss=args.bow_loss,
        encoder=args.edit_encoder,
        num_classes=args.num_classes,
    ).to("cuda")

    if args.load_model:
        pre_trained_model_path = os.path.join(args.load_model, "model.pth.best")
        pre_trained_model = torch.load(pre_trained_model_path)
        print("Loaded", pre_trained_model_path)

        model.encoder = pre_trained_model.encoder
        model.edit_encoder = pre_trained_model.edit_encoder

    if args.data_parallel:
        model = nn.DataParallel(model)
    model_save_path = os.path.join(results_path, "model.pth")
    state_dict_save_path = os.path.join(results_path, "state_dict.pth")

    train_output_save_path = os.path.join(results_path, "train.output.json")
    valid_output_save_path = os.path.join(results_path, "valid.output.json")
    test_output_save_path = os.path.join(results_path, "test.output.json")

    tmp_train_output_save_path = "/tmp/train.output.json"
    tmp_valid_output_save_path = "/tmp/valid.output.json"

    writer = SummaryWriter(results_path)

    dic_args = vars(args)
    header = "parameter|value\n - | -\n"
    parameters_string = header + "\n".join(
        [f"{key}|{value}" for key, value in dic_args.items()]
    )
    writer.add_text("parameters", parameters_string, 0)

    # ---------------------------- TRAIN -----------------------------------------------

    best_valid_loss = float("inf")
    best_valid_bleu = 0
    best_valid_meteor = 0
    best_valid_meant = 0

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

    def save_best():
        subprocess.call(
            ["cp", state_dict_save_path, state_dict_save_path + ".best"]
        )
        subprocess.call(["cp", model_save_path, model_save_path + ".best"])
        subprocess.call(
            ["cp", valid_output_save_path, valid_output_save_path + ".best"]
        )

    Optimizer = optimizers[args.optim]
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(params, args.lr)

    scheduler = Scheduler(
        optimizer,
        mode="min" if args.metric == "loss" else "max",
        factor=args.decay,
        patience=args.patience,
        threshold=0.0001,
        threshold_mode="abs",
        min_lr=1e-04,
    )

    if args.edit_encoder == "yin":
        anneal_function = get_kl_annealing_function(
            args.anneal_function, k=args.k, x0=args.x0
        )
    else:
        anneal_function = None

    print(model)
    with tqdm.trange(args.epochs, desc=model_id) as pbar:
        counter = 0
        for epoch in pbar:

            total_train_loss = 0
            batch_i = 0
            gen_sequences = []
            src_sequences = []
            src_tag_sequences = []
            tgt_sequences = []
            input_ids = []
            progress_bar = tqdm.tqdm(train_batches, desc="Loss(XXXXXXX)")
            for batch in progress_bar:
                counter += 1
                model.train()
                model.zero_grad()

                src_batch_sequences = batch.src_batch.sequences
                src_batch_tag_sequences = batch.src_batch.tag_sequences
                tgt_output_batch_sequences = batch.tgt_input_batch.sequences[
                    :, 1:
                ]
                batch_size = src_batch_sequences.shape[0]

                try:
                    batch.to_torch(device=device)
                    model_out, _ = model.forward(
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
                        changed=batch.changed.sequences,
                        changed_mask=batch.changed.masks,
                        added_edit_batch_tuple=batch.added_batch,
                        removed_edit_batch_tuple=batch.removed_batch,
                    )

                    loss, weighted_kl_loss, kl_weight = calculate_vae_loss(
                        batch_size,
                        model_out["loss"],
                        model_out["KLD"],
                        counter,
                        anneal_function,
                    )

                    progress_bar.set_description_str(
                        desc=f"Loss({str(loss.item())[:7]})"
                    )
                    writer.add_scalar(
                        "edit_vae/KL Loss (unweighted)",
                        model_out["KLD_item"] / batch_size,
                        counter,
                    )
                    writer.add_scalar(
                        "edit_vae/KL Loss (weighted)",
                        weighted_kl_loss / batch_size,
                        counter,
                    )
                    writer.add_scalar(
                        "edit_vae/NLL Loss",
                        model_out["loss"]["recon"].item() / batch_size,
                        counter,
                    )
                    writer.add_scalar(
                        "edit_vae/ppl", model_out["PPL"].item(), counter,
                    )
                    writer.add_scalar(
                        "edit_vae/BoWLoss",
                        model_out["loss"]["bow"].item() if args.bow_loss else 0,
                        counter,
                    )
                    writer.add_scalar(
                        "edit_vae/total loss", loss.item(), counter
                    )
                    writer.add_scalar("edit_vae/KL Weight", kl_weight, counter)
                    if counter % 10 == 0:
                        write_str = (
                            "Total Loss: "
                            + colors.colorize(f": {loss.item():.7f}", "cyan")
                            + " KL Loss: "
                            + colors.colorize(
                                f"{model_out['KLD_item']/batch_size:.7f}",
                                "green",
                            )
                            + " Recon Loss: "
                            + colors.colorize(
                                f"{model_out['loss']['recon'].item()/batch_size:.7f}",
                                "red",
                            )
                            + " PPL: "
                            + colors.colorize(
                                f"{model_out['PPL'].item():.7f}", "green"
                            )
                        )
                        tqdm.tqdm.write(write_str)
                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip
                    )
                    if counter == 1:
                        print(
                            "Parameters:",
                            sum([p.numel() for p in model.parameters()]),
                        )
                    optimizer.step()

                except RuntimeError as e:
                    # catch out of memory exceptions during fwd/bck (skip batch)
                    if "out of memory" in str(e):
                        pbar.write(
                            "| WARNING: ran out of memory, skipping batch. "
                            "if this happens frequently, decrease batch_size or "
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
                # print(gen_sequence_i)
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

            total_train_loss = 1.0 * total_train_loss / batch_i

            if args.device == "cuda":
                torch.cuda.empty_cache()

            output = generate_output(
                input_ids,
                src_sequences,
                tgt_sequences,
                gen_sequences,
                src_tag_sequences=src_tag_sequences,
            )

            with open(train_output_save_path, "w") as f:
                json.dump(output, f)
            with open(tmp_train_output_save_path, "w") as f:
                json.dump(output, f)

            gold_sequences = {item["id"]: item["tgt"] for item in output}
            pred_sequences = {item["id"]: item["gen"] for item in output}
            src_sequences = {item["id"]: item["src"] for item in output}

            train_bleu = bleu(gold_sequences, pred_sequences)
            train_gleu = gleu(gold_sequences, src_sequences, pred_sequences)

            # ------------------------- VALID ------------------------------------------------

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
            for batch in tqdm.tqdm(valid_batches, desc="Validation... "):
                valid_counter += 1
                model.eval()
                src_batch_sequences = batch.src_batch.sequences
                src_batch_tag_sequences = batch.src_batch.tag_sequences
                tgt_output_batch_sequences = batch.tgt_input_batch.sequences[
                    :, 1:
                ]

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
                        beam_size=args.beam_size,
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
                        counter,
                        anneal_function,
                    )

                    progress_bar.set_description_str(
                        desc=f"Loss({str(loss.item())[:7]})"
                    )
                    write_str = (
                        "Total Loss: "
                        + colors.colorize(f": {loss.item():.7f}", "cyan")
                        + " KL Loss: "
                        + colors.colorize(
                            f"{model_out['KLD_item']:.7f}", "green"
                        )
                        + " Recon Loss: "
                        + colors.colorize(
                            f"{model_out['loss']['recon'].item():.7f}", "red"
                        )
                        + " PPL: "
                        + colors.colorize(
                            f"{model_out['PPL'].item():.7f}", "green"
                        )
                    )

                    tqdm.tqdm.write(write_str)

                    total_valid_loss += loss.item()
                    total_valid_nllloss += (
                        model_out["loss"]["recon"].item() / batch_size
                    )
                    total_valid_klloss += model_out["KLD_item"] / batch_size
                    total_valid_ppl += model_out["PPL"].item()
                except RuntimeError as e:
                    # catch out of memory exceptions during fwd/bck (skip batch)
                    if "out of memory" in str(e):
                        pbar.write(
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
            if args.device == "cuda":
                torch.cuda.empty_cache()

            output = generate_output(
                input_ids,
                src_sequences,
                tgt_sequences,
                gen_sequences,
                src_tag_sequences=src_tag_sequences,
            )

            with open(valid_output_save_path, "w") as f:
                json.dump(output, f)
            with open(tmp_valid_output_save_path, "w") as f:
                json.dump(output, f)

            gold_sequences = {item["id"]: item["tgt"] for item in output}
            pred_sequences = {item["id"]: item["gen"] for item in output}
            src_sequences = {item["id"]: item["src"] for item in output}

            valid_bleu = bleu(gold_sequences, pred_sequences)
            valid_gleu = gleu(gold_sequences, src_sequences, pred_sequences)

            try:
                current_log = {
                    "Epoch": epoch,
                    "Train/Loss": total_train_loss,
                    "Train/BLEU": train_bleu,
                    "Valid/Loss": total_valid_loss,
                    "Valid/NLLLoss": total_valid_nllloss,
                    "Valid/KLLoss (UW)": total_valid_klloss,
                    "Valid/BLEU": valid_bleu,
                    "Valid/GLEU": valid_gleu,
                    "Valid/PPL": total_valid_ppl,
                }
            except AttributeError:
                current_log = {
                    "Epoch": epoch,
                    "Train/Loss": total_train_loss,
                    "Train/BLEU": train_bleu,
                    "Valid/Loss": total_valid_loss,
                    "Valid/NLLLoss": total_valid_nllloss,
                    "Valid/KLLoss (UW)": total_valid_klloss,
                    "Valid/BLEU": valid_bleu,
                    "Valid/GLEU": valid_gleu,
                    "Valid/PPL": total_valid_ppl,
                }

            pbar.write("Epoch {} ".format(epoch) + "#" * 22)
            for key, value in current_log.items():
                if key != "Epoch":
                    writer.add_scalar(f"edit_vae/{key}", float(value), epoch)
                    pbar.write(f"{key}: {value:14.3f}")
            pbar.write("\n")

            logger.update_results(current_log)

            torch.save(model, os.path.join(results_path, "model.pth"))
            torch.save(model.state_dict(), state_dict_save_path)

            if args.metric == "bleu":
                metric = current_log["Valid/BLEU"]

            elif args.metric == "loss":
                metric = current_log["Valid/Loss"]

            elif args.metric == "recon":
                metric = current_log["Valid/NLLLoss"]

            else:
                raise NotImplementedError
            stop = False
            is_best, new_lrs = scheduler.step(metric, epoch)

            if is_best:
                best_metric = metric
                datadict = {"best_metric": f"{best_metric:.4f}_{epoch}"}
                logger.update_results(datadict)
                save_best()
            else:
                pass
            for i, new_lr in enumerate(new_lrs):
                pbar.write(
                    f"Reduced learning rate of group {i} to {new_lr:.4e}."
                )
                # if new_lr <= 1e-4:
                #     stop = True
            if stop:
                break
