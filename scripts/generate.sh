gpu=$1
load_path=$2
ARGS=${@:3}

for dataset in $ARGS
do
    python generate.py --dataset $dataset --load-path $load_path
done
