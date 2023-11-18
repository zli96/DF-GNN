"""
[A Generalization of Transformer Networks to Graphs]
(https://arxiv.org/abs/2012.09699)
"""

import argparse
import pdb
import time

import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.optim as optim

from dgl.dataloading import GraphDataLoader

from dgNN.layers import choose_Inproj, preprocess_Hyper_fw_bw, SparseMHA_fused
from dgNN.utils import load_dataset_fn, parser_argument
from ogb.graphproppred import collate_dgl, Evaluator


def average(list: list) -> float:
    return sum(list) / len(list)


def check_correct(logits, logits_fuse):
    check_same = torch.tensor(
        [all(i) for i in torch.isclose(logits, logits_fuse, atol=0.01)]
    )
    if all(check_same):
        print("the results are the same, success!!!!!!!!!!")
    else:
        print("check fail!!!!!!!!!!!")
        pdb.set_trace()
        exit()


class GTModel(nn.Module):
    def __init__(
        self,
        dataset_name,
        out_size,
        hidden_size=64,
        num_layers=8,
        num_heads=1,
    ):
        super().__init__()
        self.in_proj = choose_Inproj(dataset_name, hidden_size)
        self.layers = nn.ModuleList(
            [
                SparseMHA_fused(hidden_size, hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.pooler = dglnn.SumPooling()
        print(hidden_size, out_size)
        self.predictor = nn.Linear(hidden_size, out_size)

    def forward(self, g, X, params, fuse=False):
        h = self.in_proj(X)
        # for layer in self.layers:
        #     h = layer(params, h, fuse)
        h = self.pooler(g, h)

        return self.predictor(h)


@torch.no_grad()
def evaluate(model, dataloader, device, fuse_flag):
    model.eval()
    y_true = []
    y_pred = []
    start = time.time()
    for batched_g, labels in dataloader:
        batched_g, labels = batched_g.to(device), labels.to(device)
        params = preprocess_Hyper_fw_bw(batched_g)
        ## fuse
        y_hat = model(
            batched_g,
            batched_g.ndata["feat"],
            params,
            fuse=fuse_flag,
        )
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    eval_time = time.time() - start
    print(f"evaluate time {eval_time:.2f}")
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return Evaluator("ogbg-molhiv").eval(input_dict)["rocauc"]


def check_grad(model, dataset, device, args):
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    loss_fcn = nn.BCEWithLogitsLoss()

    model.train()
    total_loss = 0.0
    for iter, (batched_g, labels) in enumerate(train_dataloader):
        batched_g, labels = batched_g.to(device), labels.to(device)
        params = preprocess_Hyper_fw_bw(batched_g)
        ## nofuse
        logits = model(batched_g, batched_g.ndata["feat"], params)
        loss = loss_fcn(logits, labels.float())
        model.zero_grad()
        loss.backward()

        q_grad = model.layers[0].q_proj.weight.grad
        k_grad = model.layers[0].k_proj.weight.grad
        v_grad = model.layers[0].v_proj.weight.grad

        ## fuse
        logits = model(batched_g, batched_g.ndata["feat"], params, fuse=True)
        loss = loss_fcn(logits, labels.float())
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()

        print(f"iter {iter} check backward correct")
        print("V grad")
        check_correct(v_grad, model.layers[0].v_proj.weight.grad)
        print("Q grad")
        check_correct(q_grad, model.layers[0].q_proj.weight.grad)
        print("K grad")
        check_correct(k_grad, model.layers[0].k_proj.weight.grad)


def train(model, dataset, device, args, fuse_flag):
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)
    loss_fcn = nn.BCEWithLogitsLoss()
    epoch_times = []
    for epoch in range(num_epochs):
        torch.cuda.synchronize()
        start = time.time()
        model.train()
        total_loss = 0.0
        for iter, (batched_g, labels) in enumerate(train_dataloader):
            batched_g, labels = batched_g.to(device), labels.to(device)
            params = preprocess_Hyper_fw_bw(batched_g)
            ## fuse
            logits = model(
                batched_g,
                batched_g.ndata["feat"],
                params,
                fuse=fuse_flag,
            )
            loss = loss_fcn(logits, labels.float())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        epoch_time = time.time() - start
        if epoch > 0:
            epoch_times.append(epoch_time)
            print(
                f"epoch {epoch:03d} time {epoch_time:.2f} avg epoch time {average(epoch_times):.2f}"
            )
        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        train_metric = evaluate(
            model,
            train_dataloader,
            device,
            fuse_flag,
        )
        print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, " f"Val: {train_metric:.4f}")


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="DOTGAT")
    parser.add_argument("--checkgrad", action="store_true")
    parser.add_argument("--num-layers", type=int, default=8)
    args = parser_argument(parser)

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset, train_fn = load_dataset_fn(args.dataset, args.data_dir)

    # Create model.
    model = GTModel(
        dataset_name=args.dataset,
        out_size=1,
        hidden_size=args.dim,
        num_heads=args.heads,
        num_layers=args.num_layers,
    ).to(dev)
    print(model)

    # Kick off training.
    if args.checkgrad:
        check_grad(model, dataset, dev, args)
    else:
        train(model, dataset, dev, args, True)
        print("---------------non-fused--------------")
        train(model, dataset, dev, args, False)
