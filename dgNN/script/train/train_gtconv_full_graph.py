import argparse

import time

import dgl
import GPUtil
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgNN.layers import preprocess_Hyper_fw_bw, SparseMHA_fused
from dgNN.utils import load_data_full_graph


class Net(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
    ):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # input projection (no residual)
        self.input_proj = nn.Linear(in_dim, num_hidden)
        self.output_proj = nn.Linear(num_hidden, num_classes)

        # hidden layers
        for l in range(num_layers):
            self.layers.append(SparseMHA_fused(num_hidden, num_hidden, 1))

    def forward(self, params, h, fuse=False):
        # h = self.input_proj(h)
        for l in range(self.num_layers):
            h = self.layers[l](params, h, fuse)
        return F.log_softmax(self.output_proj(h), dim=-1)


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def main(args):
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = load_data_full_graph(args.dataset, args.data_dir)
    if args.dataset == "arxiv":
        g = dataset[0][0]
        g.ndata["feat"] = g.ndata["feat"][:, : args.dim]
        g = g.to(dev)
        labels = dataset[0][1].squeeze(1).to(dev)
    else:
        g = dataset[0]
        g.ndata["feat"] = g.ndata["feat"][:, : args.dim]
        g = g.to(dev)
        labels = g.ndata["label"]

    # Create the sparse adjacency matrix A.
    params = preprocess_Hyper_fw_bw(g, True)
    features = g.ndata["feat"]
    print(features.shape)

    n_feats = features.shape[1]
    n_classes = dataset.num_classes

    model = Net(
        args.n_layers,
        n_feats,
        args.dim,
        n_classes,
    ).to(dev)
    if args.dataset == "arxiv":
        loss_fcn = F.nll_loss
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("----train------")
    print("----nofused------")
    for epoch in range(3):
        logits = model(params, features)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        logits = model(params, features)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")
    torch.cuda.synchronize()
    end = time.time()
    train_time = (end - start) / args.n_epochs
    print(f"no-fused avg train time {train_time*1000:.4f}")

    print("----fused------")
    for epoch in range(3):
        logits = model(params, features, fuse=True)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        logits = model(params, features, fuse=True)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")
    torch.cuda.synchronize()
    end = time.time()
    train_time_fused = (end - start) / args.n_epochs
    print(f"fused avg train time {train_time_fused*1000:.4f}")

    print("----infer------")
    print("----nofused------")
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(params, features)
    torch.cuda.synchronize()
    end = time.time()
    inference_time = (end - start) / args.n_epochs
    print(f"no-fused avg infer time {inference_time*1000:.4f}")

    print("----fused------")
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(params, features, fuse=True)
    torch.cuda.synchronize()
    end = time.time()
    inference_time_fused = (end - start) / args.n_epochs
    print(f"fused avg infer time {inference_time_fused*1000:.4f}")

    # print(f"max memory:{maxMemory}MB")
    # print("train time:", train_time)
    # print("inference time:", inference_time)

    # if args.output != None:
    #     with open("{}".format(args.output), "a") as f:
    #         print(
    #             "train_GAT_dgnn,{} heads={} hidden_dim={},{:f}s,{:f}s,{}MB,{}".format(
    #                 args.dataset,
    #                 args.n_heads,
    #                 args.n_hidden,
    #                 train_time,
    #                 inference_time,
    #                 maxMemory,
    #                 acc,
    #             ),
    #             file=f,
    #         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data-dir", type=str)

    parser.add_argument(
        "--n-epochs", type=int, default=20, help="number of training epochs"
    )
    parser.add_argument(
        "--dim", type=int, default=64, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=4, help="number of hidden layers"
    )
    # parser.add_argument("--output", type=str, default=None, help="output file")

    args = parser.parse_args()
    print(args)
    main(args)
