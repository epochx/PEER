#!/bin/bash

# for csv use:
# sqlite3 -header -csv runs.db 'SELECT

# for pretty-printed query use
# sqlite3 --header --column runs.db 'SELECT

sqlite3 --header runs.db '
  SELECT
    --run_datetime as date,
    SUBSTR(hash, 0, 8) as hash,
    --SUBSTR("commit", 0, 8) as "commit",
    --server_name,
    dataset as data,
    model_name as model,
    --seed,
    --batch_size as bz,
    --epochs,
    --lr,
    --class_lambda as lamb,
    --decay as decay,
    --num_classes as nc,
    use_kl as kl,
    ROUND("Train/Loss", 2) AS t_loss,
    ROUND("Train/BLEU", 2) as t_bleu,
    ROUND("Valid/Loss", 2) AS v_loss,
    ROUND("Valid/BLEU", 2) as v_bleu,
    ROUND("Valid/GLEU", 2) as v_gleu,
    ROUND("Valid/NLLLoss", 2) AS v_nll,
    ROUND("Valid/KLLoss (UW)", 2) as v_kl,
    ROUND("Test/Loss", 2) AS t_loss,
    ROUND("Test/BLEU", 2) as t_bleu
FROM runs' | column -n -t -s "|"