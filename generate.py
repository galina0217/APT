import argparse
import os
import time

import dgl
import numpy as np
import tensorboard_logger as tb_logger
import torch
import time
from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear


def test_moco(train_loader, model, opt):
    """
    one epoch training for moco
    """

    model.eval()

    emb_list = []
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q.to(opt.device)
        graph_k.to(opt.device)

        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)

        assert feat_q.shape == (bsz, opt.hidden_size)
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


def main(args_test):
    print("test_path", args_test.load_path)
    print("dataset", args_test.dataset)
    print("#", args_test.num_workers)
    if os.path.isfile(args_test.load_path):
        print("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]
    print(args)
    assert args_test.gpu is None or torch.cuda.is_available()
    print("Use GPU: {} for generation".format(args_test.gpu))
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)

    if args_test.dataset in GRAPH_CLASSIFICATION_DSETS:
        train_dataset = GraphClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    else:
        train_dataset = NodeClassificationDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    args.batch_size = len(train_dataset)
    print("final num_workers:", args.num_workers)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create model and optimizer
    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq,
        max_edge_freq=args.max_edge_freq,
        max_degree=args.max_degree,
        freq_embedding_size=args.freq_embedding_size,
        degree_embedding_size=args.degree_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        num_step_set2set=args.set2set_iter,
        num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model,
        norm=args.norm,
        degree_input=True,
    )

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    del checkpoint

    emb = test_moco(train_loader, model, args)
    print(args.model_folder)
    np.save(os.path.join(args.model_folder, args_test.dataset), emb.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    # fmt: off
    parser.add_argument("--load-path",
                        default="saved/apt/current.pth")
    parser.add_argument("--dataset", type=str, default="imdb-binary",
                        choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport",
                                 "cora", "citeseer", "pubmed", "squirrel", "texas", "cornell", "wisconsin", "youtube",
                                 "flickr", "blogcatalog", "kdd", "chameleon", "icdm", "chameleon", "wiki", "facebook",
                                 "sigir", "cikm", "sigmod", "icde", "cs", "phy", "cora_full", "photo", "computer",
                                 "h-index-rand-1", "h-index-top-1", "h-index", "polblogs", "DD242", "DD68", "DD687",
                                 "academia", "p2p25",
                                 "gene"] + GRAPH_CLASSIFICATION_DSETS)
    parser.add_argument("--gpu", default="0", type=int, help="GPU id to use.")
    parser.add_argument("--num-workers", type=int, default=1, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=1, help="num of dataset copies that fit in memory")
    # fmt: on
    t1 = time.time()
    main(parser.parse_args())
    t2 = time.time()
    print("total time:", t2 - t1)
