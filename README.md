# Better with Less: A Data-Active Perspective on Pre-Training Graph Neural Networks

## About

This project is the implementation of paper "Better with Less: A Data-Active Perspective on Pre-Training Graph Neural Networks"

## Dependencies
The script has been tested running under Python 3.7.10, with the following packages installed (along with their dependencies):

- [PyTorch](https://pytorch.org/). Version >=1.4 required. You can find instructions to install from source [here](https://pytorch.org/get-started/previous-versions/).
- [DGL](https://www.dgl.ai/). 0.5 > Version >=0.4.3 required. You can find instructions to install from source [here](https://www.dgl.ai/pages/start.html).
- [rdkit](https://anaconda.org/conda-forge/rdkit). Version = 2019.09.2 required. It can be easily installed with 
			```conda install -c conda-forge rdkit=2019.09.2```
- [Other Python modules](https://pypi.python.org). Some other Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

	`pip install -r requirements.txt`

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


## File folders

`dataset`: contains the data of "DD242, DD68, DD687".

`dataset/splits`: **need to unzipped**, contains the split data of "Cora, Pubmed, Cornell and Wisconsin".

`scripts`: contains all the scripts for running code.

`gcc&utils`: contains the code of models.

## Usage: How to run the code
We divide it into two steps (1) Pre-training/Finetuning (2) Evaluating.

### 1. Pre-training / Fine-tuning

Before running the actual pretraining commands, the code requires you to download the dataset. Pre-training datasets is stored in `data.bin`. And the datasets can be download through [website](https://drive.google.com/file/d/1kbOciSHXSOAFV7X1CuL_nm9_2sxKeDfU/view).

**1.1 Pretraining**


```bash
python train_apt.py \
  --model-path <saved file> \
  --threshold <uncertainty threshold for choosing samples >
  --tb-path <tensorboard file> \
  --dgl_file <dataset in bin format> \
  --moco
```
For more detail, the help information of the main script `train_apt.py` can be obtain by executing the following command.

```bash
python train_apt.py -h

optional arguments:
  --max_period MAX_PERIOD
                        maximal period (default:6)
  --threshold THRESHOLD
                        uncertainty threshold for choosing samples (default:3)
  --threshold_g THRESHOLD_G
                        threshold for moving to a new graph (default:3.5)
  --lay LAY             layer number
  --ewc_rate EWC_RATE   weight for regularization (default:10)
  --regular REGULAR     whether there is regularization (default:True)
  --random RANDOM       whether there is a random selection (default:False)
  --init INIT           init epoch for training (default:20)
  --select-criterion SELECT_CRITERION
                        the criterion for graph selector (all, entropy,
                        degstd, density, degmean, alpha)(default:all)
  --intercept INTERCEPT
                        intercept for beta distribution(default:3)
  --ewc EWC             whether to use ewc loss (default:False)

```

**Demo:**	

```bash
python train_apt.py \
  --max_period 6 \
  --threshold 3 \
  --threshold_g 3.5 \
  --lay 3 \
  --ewc_rate 10 \
  --regular 1 \
  --model-path saved \
  --tb-path tensorboard  \
  --dgl_file data.bin \
  --moco 
```

**1.2 Fine-tuning**


To Finetune APT on all downstream datasets:

```
bash scripts/evaluate_generate.sh <saved file>
```

**Demo:**

```
bash scripts/evaluate_generate.sh saved
```

### 2. Evaluating

`generate.py` file helps generate embeddings on a specific dataset. The help information of the main script `generate.py` can be obtain by executing the following command.

```bash
python generate.py -h

optional arguments:
  --load-path LOAD_PATH
  --dataset Dataset
  --gpu GPU  GPU id to use.
```
The embedding will be used for evaluation in node classification and graph classification. The script `evaluate.sh` and `evaluate_finetune.sh` are available to simplify the evaluation process.

**2.1 Evaluate without fine-tuning on all downstream datasets:**

```
bash evaluate.sh <load path>
```


**Demo:**

```
bash scripts/evaluate.sh saved
```


**2.2 Evaluate after fine-tuning on all downstream datasets:**

```
bash evaluate_finetune.sh <load path>
```

**Demo:**

```
bash scripts/evaluate_finetune.sh saved
```

## Citing APT

If you use APT in your research or wish to refer to the baseline results, please use the following BibTeX.

```
@inproceedings{
xu2023better,
title={Better with Less: A Data-Centric Prespective on Pre-Training Graph Neural Networks},
author={Jiarong Xu, Renhong Huang, Xin Jiang, Yuxuan Cao, Carl Yang, Chunping Wang, Yang Yang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
}
```

## Acknowledgements
Part of this code is inspired by Qiu et al.'s [GCC: Graph Contrastive Coding](https://github.com/THUDM/GCC).

