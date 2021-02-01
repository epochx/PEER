# Variational Inference for Learning Representations of Natural Language Edits

Code accompanying the paper [Variational Inference for Learning Representations of Natural Language Edits](https://arxiv.org/abs/2004.09143).


## Installation

1. Clone this repository.

    ```bash
    git clone https://github.com/epochx/PEER
    cd PEER
    ```

2. We recommend conda for package management. Create a conda environment and activate it.

   ```bash
   conda create -n <name> python=3.6
   conda activate <name>
   ```

   Where you can replace `<name>` by whatever you want.

3. Setup everything
   ```bash
   sh install.sh
   ```
   This script will do the following:
   * Install all the dependencies to the currently-active conda environment.
   * Download and unzip our data and splits.
   

Downloading and preprocessing everything can take a while depending on your Internet connection and your conda cache. By default, the data will be placed in the `~/data/PEER` folder, and results will be stored on the `~/results/PEER` folder. To use different locations, simply move the files to the desired paths and change the contents of `settings.py` accordingly.

## Training Instructions

To replicate experiments in our paper, run the training script.

```bash
python train_vae.py --config <path/to/config/file>
```

Where `<path/to/config/file>` should point at a yaml file like the ones located in the `experiments` folder, where we provide our exact experimental configurations. These should allow you to replicate the results in our paper. Please note that some of our models require existing pre-trained models, since we initialize the parameters our variational auto-encoder with the weights of a regular auto-encoder for improved convergence and performance. 

## Evaluation Instructions

### Intrinsic evaluation

Given a trained model, use the following command to perform the intrinsic evaluation.

```bash
python test_vae.py --model <path/to/pretrained/model/folder> --best
```

This will attempt to load the best model checkpoint in the folder `<path/to/pretrained/model/folder>` and run the intrinsic evaluation on the test set of the same dataset where the model was trained. For additional details on how to run this command, please check `python test_vae.py --help`.

### Extracting Edit Representations and Extrinsic Evaluation

It is possible to extract edit representations produced by a trained model using the following command.

```bash
python encode.py --model <path/to/pretrained/model/folder> --batch_size <batch_suize> --split all --best --dataset <dataset>
```

This will attempt to load the best performing model stored in the folder `<path/to/pretrained/model/folder>`, and extract edit representations of the examples in the provided dataset, using the data in all the splits. These representations will be stored in a json file that will be placed on the same folder where the provided model was located, using an adequate filename. In an upcoming update we will also release our pre-trained models alongside their produced edit representations. 

Finally, you can run the extrinsic evaluation on the downstream tasks for the extracted edit representations by running the following command.

```bash
python train_multiclass_sklearn.py --<path/to/json/file> --num_layers <num_layers>
```

Where `<path/to/json/file>` points at the json file containing the extracted edit representations, and `<num_layers>` is the depth of the classifier model (0, 1 or 2 are supported). When running this command, make sure you provide a json file that has been created using all the splits for a given dataset. In the case of training on the `QT21 De-EN MQM` dataset, please use `python --train_multilabel_sklearn.py --<path/to/json/file>` instead, as this downstream task is a multi-label classification one.

## Data  

If you are just interested in the datasets, our data and splits are available on this [link](https://zenodo.org/record/4478267). You can directly download everything by running the following command.

```bash
wget https://zenodo.org/record/4478267/files/PEER.zip
```

## Citation

If you use this code please consider citing our paper.

```tex
@inproceedings{marresetaylor2021variational,
  title     = {Variational {{Inference}} for {{Learning Representations}} of {{Natural Language Edits}}},
  author    = {Marrese-Taylor, Edison and Reid Machel and Matsuo, Yutaka},
  year      = {2021},
  month     = {February},
  publisher = {AAAI Press},
  booktitle = {Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (To appear)},
  series    = {AAAI'21}
}
```

## To-dos

- [ ] Upload pre-trained models and extracted edit representations
- [ ] Add instructions to generate our datasets (either from the corresponding original ones, or from scratch). 

