#!/bin/bash
saved_path=$1
cuda=$2
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_cora/current.pth" --dataset cora
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cora/" 64 cora
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_pubmed/current.pth" --dataset pubmed
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_pubmed/" 64 pubmed
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_brazil_airport/current.pth" --dataset brazil_airport
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_brazil_airport/" 64 brazil_airport
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD242/current.pth" --dataset DD242
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD242/" 64 DD242
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD68/current.pth" --dataset DD68
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD68/" 64 DD68
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_DD687/current.pth" --dataset DD687
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_DD687/" 64 DD687
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_wisconsin/current.pth" --dataset wisconsin
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_wisconsin/" 64 wisconsin
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_cornell/current.pth" --dataset cornell
bash scripts/node_classification/ours.sh "./$saved_path/apt_finetune_cornell/" 64 cornell
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_imdb-binary/current.pth" --dataset imdb-binary
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_imdb-binary/" 64 imdb-binary
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_dd/current.pth" --dataset dd
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_dd/" 64 dd
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc
python generate.py --load-path "./$saved_path/apt_finetune_msrc/current.pth" --dataset msrc
bash scripts/graph_classification/ours.sh "./$saved_path/apt_finetune_msrc/" 64 msrc