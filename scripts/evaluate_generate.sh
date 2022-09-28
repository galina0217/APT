#!/bin/bash
saved_path=$1
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset cora --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset brazil_airport --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset DD242 --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset DD68 --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset DD687 --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset wisconsin --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset cornell --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset imdb-binary --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset dd --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset msrc --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
python train_al.py --exp FT --model-path "./$saved_path" --tb-path tensorboard --tb-freq 5 --dataset pubmed --finetune --epochs 50 --resume "./$saved_path/GCC_perturb_16384_0.001_self/current.pth"
