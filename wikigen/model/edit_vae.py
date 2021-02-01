import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .loss import BoWLoss
from .decoders import EditDecoder
from .encoders import LSTM_VAEncoder, LSTM_Encoder, GuuEncoder
from .beams import beam_search_decode


class EditVAE(nn.Module):
    def __init__(
        self,
        embeddings,
        encoder_hidden,
        decoder_hidden,
        dropout: dict,
        latent_size: int,
        latent_out=None,
        encoder_num_layers=1,
        decoder_num_layers=1,
        num_classes=None,
        use_kl=False,
        use_pointer=False,
        bow_loss=False,
        encoder="yin",
        kappa=10,
    ):
        super(EditVAE, self).__init__()
        if use_pointer:
            raise NotImplementedError
        self.embeddings = embeddings
        self.categorical_dim = 1
        self.bow_loss_bool = bow_loss

        if encoder == "yin":
            self.edit_encoder = LSTM_VAEncoder(
                embeddings["edit_before"],
                embeddings["edit_after"],
                embeddings["tags"],
                encoder_hidden,
                num_layers=encoder_num_layers,
                input_dropout=dropout["edit"]["input"],
                output_dropout=dropout["edit"]["output"],
            )
        elif encoder == "guu":
            self.edit_encoder = GuuEncoder(
                embeddings["edit_before"], latent_size, kappa, use_kl=use_kl
            )
        else:
            raise NotImplementedError

        self.encoder = LSTM_Encoder(
            embeddings["before"],
            encoder_hidden,
            num_layers=encoder_num_layers,
            input_dropout=dropout["before"]["input"],
            output_dropout=dropout["before"]["output"],
            word_dropout=dropout["before"]["word"],
        )

        self.decoder = EditDecoder(
            embeddings["after"],
            decoder_hidden,
            repr_hidden_size=latent_out
            if latent_out is not None
            else decoder_hidden,
            num_layers=decoder_num_layers,
            input_dropout=dropout["after"]["input"],
            output_dropout=dropout["after"]["output"],
            word_dropout=dropout["after"]["word"],
            teacher_forcing_p=0.5,
            attention="general",
        )

        self.latent_size = latent_size
        self.use_kl = use_kl

        if self.use_kl:
            if encoder == "yin":
                self.mean_layer = nn.Linear(
                    self.encoder.hidden * 2, latent_size
                )
                self.logvar_layer = nn.Linear(
                    self.encoder.hidden * 2, latent_size
                )
                self.latent2hidden = nn.Linear(latent_size, decoder_hidden)
                self.combo_layer = nn.Linear(
                    sum((encoder_hidden * 2, decoder_hidden)), decoder_hidden
                )
            else:
                self.latent2hidden = nn.Linear(latent_size, decoder_hidden)
                self.combo_layer = nn.Linear(
                    sum((encoder_hidden * 2, decoder_hidden)), decoder_hidden
                )

        else:
            if encoder == "guu":
                raise NotImplementedError

            self.hidden2latent_layer = nn.Linear(
                self.encoder.hidden * 2, latent_size
            )
            self.latent2hidden = nn.Linear(latent_size, decoder_hidden)
            self.combo_layer = nn.Linear(encoder_hidden * 4, decoder_hidden)

        if self.bow_loss_bool:
            self.bow_loss = BoWLoss(
                self.latent_size, self.embeddings["after"].num_embeddings
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def hidden2latent(self, hidden_state, temp=None):
        mean = self.mean_layer(hidden_state)
        logvar = self.logvar_layer(hidden_state)

        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    def forward(
        self,
        before_edit,
        after_edit,
        tags,
        before,
        after,
        seq_lens,
        src_batch_mask,
        tgt_batch_mask,
        changed,
        changed_mask,
        added_edit_batch_tuple=None,
        removed_edit_batch_tuple=None,
    ):
        ## Edit encoding and reparameterization

        batch_size = before_edit.size(0)
        bow_loss = 0
        if isinstance(self.edit_encoder, LSTM_VAEncoder):
            edit_enc_hidden, _ = self.edit_encoder.forward(
                before_edit, after_edit, tags, seq_lens["edit"]
            )
            # now get the latent vector
            if self.use_kl:
                z, mean, logvar = self.hidden2latent(edit_enc_hidden)
            else:
                z = self.hidden2latent_layer(edit_enc_hidden)

        elif isinstance(self.edit_encoder, GuuEncoder):
            z = self.edit_encoder.forward(
                added_edit_batch_tuple, removed_edit_batch_tuple
            )
        else:
            raise NotImplementedError

        if self.bow_loss_bool:
            bow_loss = self.bow_loss.forward(z, changed, changed_mask)

        ## $edit^-$ encoding
        edit_minus_hidden, edit_minus_full_h_t = self.encoder.forward(
            before, seq_lens["before"]
        )

        # initialize decoder hidden state from latent
        dec_edit_repr = self.latent2hidden(z)
        decoder_init_state = self.combo_layer(
            torch.cat([edit_minus_hidden, dec_edit_repr], -1)
        )

        # decode
        decoder_loss, predictions, ppl = self.decoder.forward(
            seq=after,
            representation=dec_edit_repr,
            encoder_hidden_states=edit_minus_full_h_t,
            src_batch_mask=src_batch_mask,
            initial_state=decoder_init_state,
        )

        kl_loss = 0
        kl_loss_item = 0
        if self.use_kl:
            if isinstance(self.edit_encoder, LSTM_VAEncoder):
                kl_loss = self.kl_div(mean, logvar)
                try:
                    kl_loss_item = kl_loss.mean().item()
                except:
                    kl_loss_item = kl_loss.item()

        return_dict = {
            "preds": predictions,
            "loss": {"bow": bow_loss, "recon": decoder_loss,},
            "KLD": kl_loss,
            "KLD_item": kl_loss_item,
            "PPL": ppl,
        }

        return return_dict, dec_edit_repr

    def kl_div(self, mu_post=None, log_sigma_post=None):

        KLD = torch.mean(
            -0.5
            * torch.sum(
                1 + log_sigma_post - mu_post.pow(2) - log_sigma_post.exp(), -1,
            )
        )

        return KLD

    def validation(
        self,
        before_edit,
        after_edit,
        tags,
        before,
        after,
        seq_lens,
        src_batch_mask,
        tgt_batch_mask,
        changed,
        changed_mask,
        vocab,
        beam_size,
        max_len,
        added_edit_batch_tuple=None,
        removed_edit_batch_tuple=None,
    ):
        batch_size = before_edit.size(0)
        bow_loss = 0
        if isinstance(self.edit_encoder, LSTM_VAEncoder):
            edit_enc_hidden, _ = self.edit_encoder.forward(
                before_edit, after_edit, tags, seq_lens["edit"]
            )
            # now get the latent vector
            if self.use_kl:
                z, mean, logvar = self.hidden2latent(edit_enc_hidden)
            else:
                z = self.hidden2latent_layer(edit_enc_hidden)

        elif isinstance(self.edit_encoder, GuuEncoder):
            z = self.edit_encoder.forward(
                added_edit_batch_tuple, removed_edit_batch_tuple
            )
        else:
            raise NotImplementedError

        ## $edit^-$ encoding
        edit_minus_hidden, edit_minus_full_h_t = self.encoder.forward(
            before, seq_lens["before"]
        )

        # initialize decoder hidden state from latent
        dec_edit_repr = self.latent2hidden(z)
        decoder_init_state = self.combo_layer(
            torch.cat([edit_minus_hidden, dec_edit_repr], -1)
        )

        # decode
        decoder_loss, predictions, ppl = self.decoder.forward(
            seq=after,
            representation=dec_edit_repr,
            encoder_hidden_states=edit_minus_full_h_t,
            src_batch_mask=src_batch_mask,
            initial_state=decoder_init_state,
        )

        kl_loss = 0
        kl_loss_item = 0
        if self.use_kl:
            if isinstance(self.edit_encoder, LSTM_VAEncoder):
                kl_loss = self.kl_div(mean, logvar)
                try:
                    kl_loss_item = kl_loss.mean().item()
                except:
                    kl_loss_item = kl_loss.item()

        # predict
        predictions, _ = beam_search_decode(
            self,
            vocab,
            max_length=max_len,
            beam_size=beam_size,
            context_batch_mask=src_batch_mask,
            context_batch=edit_minus_full_h_t,
            z=dec_edit_repr,
            init_state=decoder_init_state,
        )

        return_dict = {
            "preds": predictions,
            "loss": {"bow": bow_loss, "recon": decoder_loss,},
            "KLD": kl_loss,
            "KLD_item": kl_loss_item,
            "PPL": ppl,
        }

        return return_dict, dec_edit_repr

    def encode(
        self,
        before_edit,
        after_edit,
        tags,
        before,
        after,
        seq_lens,
        src_batch_mask,
        tgt_batch_mask,
        return_logvar=False,
        added_edit_batch_tuple=None,
        removed_edit_batch_tuple=None,
    ):
        ## Edit encoding and reparameterization
        self.eval()
        batch_size = before_edit.size(0)
        bow_loss = 0
        if isinstance(self.edit_encoder, LSTM_VAEncoder):
            edit_enc_hidden, _ = self.edit_encoder.forward(
                before_edit, after_edit, tags, seq_lens["edit"]
            )

            if self.use_kl:
                z, mean, logvar = self.hidden2latent(edit_enc_hidden)
                if return_logvar:
                    return mean, logvar
                else:
                    return mean
            else:
                z = self.hidden2latent_layer(edit_enc_hidden)
                if return_logvar:
                    return z, None
                else:
                    return z
        elif isinstance(self.edit_encoder, GuuEncoder):
            z = self.edit_encoder.forward(
                added_edit_batch_tuple, removed_edit_batch_tuple
            )
            if return_logvar:
                return z, None
            else:
                return z

    def decode(
        self,
        before,
        seq_lens,
        src_batch_mask,
        vocab,
        beam_size,
        max_len,
        edit_representation_batch=None,
        logvar_coef=1.0,
        before_edit=None,
        after_edit=None,
        tags=None,
        after=None,
        tgt_batch_mask=None,
        added_edit_batch_tuple=None,
        removed_edit_batch_tuple=None,
    ):

        ## FIRST ENCODE edit^-
        edit_minus_hidden, edit_minus_full_h_t = self.encoder.forward(
            before, seq_lens["before"]
        )

        ## THEN ENCODE the edit if not provided
        if edit_representation_batch is None:

            if isinstance(self.edit_encoder, GuuEncoder):
                raise NotImplementedError

            assert after_edit is not None
            assert tags is not None
            assert before is not None
            assert after is not None
            assert tgt_batch_mask is not None

            batch_size = before_edit.size(0)
            bow_loss = 0
            if isinstance(self.edit_encoder, LSTM_VAEncoder):
                edit_enc_hidden, _ = self.edit_encoder.forward(
                    before_edit, after_edit, tags, seq_lens["edit"]
                )
            else:
                z = self.edit_encoder.forward(
                    added_edit_batch_tuple, removed_edit_batch_tuple
                )

            # now get the latent vector
            bow_loss = 0
            if isinstance(self.edit_encoder, LSTM_VAEncoder):
                if self.use_kl:
                    _, mean, logvar = self.hidden2latent(edit_enc_hidden)
                    z = self.reparameterize(mean, logvar * logvar_coef)
                else:
                    z = self.hidden2latent_layer(edit_enc_hidden)

        else:
            z = edit_representation_batch

        # initialize decoder and predict
        dec_edit_repr = self.latent2hidden(z)

        decoder_init_state = self.combo_layer(
            torch.cat([edit_minus_hidden, dec_edit_repr], -1)
        )

        predictions, _ = beam_search_decode(
            self,
            vocab,
            max_length=max_len,
            beam_size=beam_size,
            context_batch_mask=src_batch_mask,
            context_batch=edit_minus_full_h_t,
            z=dec_edit_repr,
            init_state=decoder_init_state,
        )

        decoder_loss = None
        ppl = None
        if edit_representation_batch is None:
            # IF edit not provided, also compute the loss
            decoder_loss, _, ppl = self.decoder.forward(
                seq=after,
                representation=dec_edit_repr,
                encoder_hidden_states=edit_minus_full_h_t,
                src_batch_mask=src_batch_mask,
                initial_state=decoder_init_state,
            )

        return_dict = {
            "preds": predictions,
            "loss": {"recon": decoder_loss},
            "PPL": ppl,
        }

        return return_dict

    def interpolate(
        self,
        before_edit,
        after_edit,
        tags,
        before,
        after,
        seq_lens,
        src_batch_mask,
        tgt_batch_mask,
        vocab,
        beam_size,
        max_len,
        num_points,
        logvar_coef=1.0,
    ):

        batch_size = before.size(0)

        assert batch_size % 2 == 0

        chunk_size = int(batch_size / 2)

        ## ENCODE edit^-
        edit_minus_hidden, edit_minus_full_h_t = self.encoder.forward(
            before, seq_lens["before"]
        )

        (
            edit_minus_hidden_source,
            edit_minus_hidden_target,
        ) = edit_minus_hidden.split(chunk_size, 0)

        (src_batch_mask_source, src_batch_mask_target) = src_batch_mask.split(
            chunk_size, 0
        )

        edit_minus_hidden_source = edit_minus_hidden_source.repeat(
            num_points + 2, 1
        )
        edit_minus_hidden_target = edit_minus_hidden_target.repeat(
            num_points + 2, 1
        )

        (
            edit_minus_full_h_t_source,
            edit_minus_full_h_t_target,
        ) = edit_minus_full_h_t.split(chunk_size, 0)

        edit_minus_full_h_t_source = edit_minus_full_h_t_source.repeat(
            num_points + 2, 1, 1
        )
        edit_minus_full_h_t_target = edit_minus_full_h_t_target.repeat(
            num_points + 2, 1, 1
        )

        src_batch_mask_source = src_batch_mask_source.repeat(num_points + 2, 1)
        src_batch_mask_target = src_batch_mask_target.repeat(num_points + 2, 1)

        # ENCODE edit
        # -> (batch_size, latent_size)
        input_mu, input_logvar = self.encode(
            before_edit,
            after_edit,
            tags,
            before,
            after,
            seq_lens,
            src_batch_mask,
            tgt_batch_mask,
            return_logvar=True,
        )

        # -> (1, latent_size)
        source_mu, target_mu = input_mu.split(chunk_size, 0)

        if self.use_kl:
            source_logvar, target_logvar = input_logvar.split(chunk_size, 0)

            source_latent = self.reparameterize(
                source_mu, source_logvar * logvar_coef
            )

            target_latent = self.reparameterize(
                target_mu, target_logvar * logvar_coef
            )
        else:
            source_latent, target_latent = source_mu, target_mu

        interpolated_latent = []
        for dim, (s, e) in enumerate(
            zip(source_latent.squeeze(), target_latent.squeeze())
        ):
            interpolated_dim = torch.linspace(
                s.item(), e.item(), num_points + 2
            )
            interpolated_latent.append(interpolated_dim)

        z = torch.stack(interpolated_latent, 1).to("cuda")

        dec_edit_repr = self.latent2hidden(z)

        decoder_init_state_source = self.combo_layer(
            torch.cat([edit_minus_hidden_source, dec_edit_repr], -1)
        )

        decoder_init_state_target = self.combo_layer(
            torch.cat([edit_minus_hidden_target, dec_edit_repr], -1)
        )

        predictions_source, _ = beam_search_decode(
            self,
            vocab,
            max_length=max_len,
            beam_size=beam_size,
            context_batch_mask=src_batch_mask_source,
            context_batch=edit_minus_full_h_t_source,
            z=dec_edit_repr,
            init_state=decoder_init_state_source,
        )

        predictions_target, _ = beam_search_decode(
            self,
            vocab,
            max_length=max_len,
            beam_size=beam_size,
            context_batch_mask=src_batch_mask_target,
            context_batch=edit_minus_full_h_t_target,
            z=dec_edit_repr,
            init_state=decoder_init_state_target,
        )

        return predictions_source, predictions_target
