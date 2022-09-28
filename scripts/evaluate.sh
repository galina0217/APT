#!/bin/bash
saved_path=$1
cuda=$2
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cora
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 pubmed
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 brazil_airport
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD242
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD68
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 DD687
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 wisconsin
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 cornell
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 imdb-binary
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 dd
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc
python generate.py --load-path "./$saved_path/GCC_perturb_16384_0.001_self/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/GCC_perturb_16384_0.001_self/" 64 msrc