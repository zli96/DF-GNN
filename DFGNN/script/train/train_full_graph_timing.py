import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from DFGNN.layers import preprocess_Hyper_fw_bw, SparseMHA_forward
from DFGNN.utils import load_data_full_graph

from tabulate import tabulate


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
        self.input_proj = nn.Linear(in_dim, num_hidden)
        self.output_proj = nn.Linear(num_hidden, num_classes)

        for l in range(num_layers):
            self.layers.append(SparseMHA_forward(num_hidden, num_hidden, 1))

    def forward(self, params, h, fuse=False):
        h = self.input_proj(h)
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
    g = dataset[0]
    g.ndata["feat"] = g.ndata["feat"][:, : args.dim]
    g = g.to(dev)
    labels = g.ndata["label"]

    # Create the sparse adjacency matrix A.
    params = preprocess_Hyper_fw_bw(g, True)
    features = g.ndata["feat"]

    n_feats = features.shape[1]
    n_classes = dataset.num_classes

    model = Net(
        args.n_layers,
        n_feats,
        args.dim,
        n_classes,
    ).to(dev)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("----Measure full training time------")

    ## warpup
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
    epoch_time = ((end - start) / args.n_epochs) * 1000
    print(f"no-fused avg train time {epoch_time:.4f}")

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
    epoch_time_fused = ((end - start) / args.n_epochs) * 1000
    print(f"fused avg train time {epoch_time_fused:.4f}")

    print("----Measure forward+backward time------")
    ## warpup
    for epoch in range(3):
        logits = model(params, features)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()

    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        logits = model(params, features)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    fw_bw_time = ((end - start) / args.n_epochs) * 1000
    print(f"no-fused avg train time {fw_bw_time:.4f}")

    ## warpup
    for epoch in range(3):
        logits = model(params, features, fuse=True)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()

    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        logits = model(params, features, fuse=True)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    fw_bw_time_fused = ((end - start) / args.n_epochs) * 1000
    print(f"fused avg train time {fw_bw_time_fused:.4f}")

    print("----Measure forward time------")
    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(params, features)
        loss = loss_fcn(logits, labels)
    torch.cuda.synchronize()
    end = time.time()
    fw_time = ((end - start) / args.n_epochs) * 1000
    print(f"no-fused avg forward time {fw_time:.4f}")

    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(params, features, fuse=True)
        loss = loss_fcn(logits, labels)
    torch.cuda.synchronize()
    end = time.time()
    fw_time_fused = ((end - start) / args.n_epochs) * 1000
    print(f"fused avg forward time {fw_time_fused:.4f}")

    bw_time = fw_bw_time - fw_time
    bw_time_fused = fw_bw_time_fused - fw_time_fused

    up_time = epoch_time - fw_bw_time
    up_time_fused = epoch_time_fused - fw_bw_time_fused

    stage_time = [
        ["DGL Sparse", fw_time, bw_time, up_time, epoch_time],
        ["DF-GNN", fw_time_fused, bw_time_fused, up_time_fused, epoch_time_fused],
    ]

    print(
        tabulate(
            stage_time,
            headers=["", "Forward(ms)", "Backward(ms)", "Update(ms)", "Sum(ms)"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data-dir", type=str)

    parser.add_argument(
        "--n-epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument(
        "--dim", type=int, default=64, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=8, help="number of hidden layers"
    )
    args = parser.parse_args()
    main(args)
