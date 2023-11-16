"""
[A Generalization of Transformer Networks to Graphs]
(https://arxiv.org/abs/2012.09699)
"""

import argparse
import pdb

import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader

from dgNN.layers import preprocess_Hyper_fw_bw, SparseMHA_fused
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm


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


def Move2Device(data_list, dev):
    ### move data in list to dev
    data_dev = []
    for data in data_list:
        if isinstance(data, tuple):
            data_dev.append(
                [param.to(dev) if hasattr(param, "to") else param for param in data]
            )
        elif hasattr(data, "to"):
            data_dev.append(data.to(dev))
        else:
            data_dev.append(data)
    return data_dev


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.MHA = SparseMHA_fused(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, params, h, fuse):
        h1 = h
        h = self.MHA(params, h, fuse=fuse)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)


class GTModel(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_size=64,
        pos_enc_size=2,
        num_layers=8,
        num_heads=1,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_size)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.pooler = dglnn.SumPooling()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
        )

    def forward(self, g, X, pos_enc, params, fuse=False):
        h = self.atom_encoder(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(params, h, fuse)
        h = self.pooler(g, h)

        return self.predictor(h)


@torch.no_grad()
def evaluate(model, dataloader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    for batched_g, labels in dataloader:
        params = preprocess_Hyper_fw_bw(batched_g)
        ## fuse
        batched_g, labels, params = Move2Device([batched_g, labels, params], device)
        y_hat = model(
            batched_g,
            batched_g.ndata["feat"],
            batched_g.ndata["PE"],
            params,
            fuse=True,
        )
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["rocauc"]


def check_grad(model, dataset, device):
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=256,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    loss_fcn = nn.BCEWithLogitsLoss()

    model.train()
    total_loss = 0.0
    for iter, (batched_g, labels) in enumerate(train_dataloader):
        params = preprocess_Hyper_fw_bw(batched_g)
        ## nofuse
        batched_g, labels, params = Move2Device([batched_g, labels, params], device)
        logits = model(
            batched_g, batched_g.ndata["feat"], batched_g.ndata["PE"], params
        )
        loss = loss_fcn(logits, labels.float())
        model.zero_grad()
        loss.backward()

        q_grad = model.layers[0].MHA.q_proj.weight.grad
        k_grad = model.layers[0].MHA.k_proj.weight.grad
        v_grad = model.layers[0].MHA.v_proj.weight.grad

        ## fuse
        batched_g, labels, params = Move2Device([batched_g, labels, params], dev)
        logits = model(
            batched_g, batched_g.ndata["feat"], batched_g.ndata["PE"], params, fuse=True
        )
        loss = loss_fcn(logits, labels.float())
        total_loss += loss.item()
        model.zero_grad()
        loss.backward()

        print(f"iter {iter} check backward correct")
        print("V grad")
        check_correct(v_grad, model.layers[0].MHA.v_proj.weight.grad)
        print("Q grad")
        check_correct(q_grad, model.layers[0].MHA.q_proj.weight.grad)
        print("K grad")
        check_correct(k_grad, model.layers[0].MHA.k_proj.weight.grad)


def train(model, dataset, evaluator, device):
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=256,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    valid_dataloader = GraphDataLoader(
        dataset[dataset.val_idx], batch_size=256, collate_fn=collate_dgl
    )
    test_dataloader = GraphDataLoader(
        dataset[dataset.test_idx], batch_size=256, collate_fn=collate_dgl
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)
    loss_fcn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for iter, (batched_g, labels) in enumerate(train_dataloader):
            params = preprocess_Hyper_fw_bw(batched_g)
            ## fuse
            batched_g, labels, params = Move2Device([batched_g, labels, params], device)
            logits = model(
                batched_g,
                batched_g.ndata["feat"],
                batched_g.ndata["PE"],
                params,
                fuse=True,
            )
            loss = loss_fcn(logits, labels.float())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        val_metric = evaluate(model, valid_dataloader, evaluator, device)
        test_metric = evaluate(model, test_dataloader, evaluator, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, "
            f"Val: {val_metric:.4f}, Test: {test_metric:.4f}"
        )


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="DOTGAT")
    parser.add_argument("--checkgrad", action="store_true")
    args = parser.parse_args()

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    pos_enc_size = 8
    dataset = AsGraphPredDataset(
        DglGraphPropPredDataset("ogbg-molhiv", "/workspace2/dataset")
    )
    evaluator = Evaluator("ogbg-molhiv")
    # laplacian positional encoding
    for g, _ in tqdm(dataset, desc="Computing Laplacian PE"):
        # g.ndata["PE"] = dgl.lap_pe(g, k=pos_enc_size, padding=True)
        g.ndata["PE"] = torch.randn((g.num_nodes(), pos_enc_size))

    # preprocess
    preprocess_func = preprocess_Hyper_fw_bw

    # Create model.
    out_size = dataset.num_tasks
    model = GTModel(out_size=out_size, pos_enc_size=pos_enc_size).to(dev)
    print(model)

    # Kick off training.
    if args.checkgrad:
        check_grad(model, dataset, dev)
    else:
        train(model, dataset, evaluator, dev)
