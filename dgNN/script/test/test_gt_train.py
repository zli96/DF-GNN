import argparse

import torch

import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from dgNN.layers import load_prepfunc, preprocess_Hyper_fw_bw, SparseMHA_hyper

from dgNN.utils import check_correct, load_dataset_fn, parser_argument


class GTModel(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, hidden_size, outsize):
        super().__init__()
        self.MHA = layer
        self.embedding_h = nn.Embedding(3, hidden_size)
        self.predict = nn.Linear(hidden_size, outsize)

    def forward(self, params, X, fuse=False):
        h = self.embedding_h(X)
        h = self.MHA(params, h, fuse)
        return self.predict(h)


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


def train(process_func, model, train_dataloader, dev, fuse_flag):
    r"""training function for the graph-level task"""
    print("----------------------Forward------------------------")
    loss_fcn = nn.BCEWithLogitsLoss()

    for i, (batched_g) in enumerate(train_dataloader):
        print(f"-----epoch {i}--------")
        ## preprocess
        params = process_func(batched_g)
        batched_g, params = Move2Device([batched_g, params], dev)
        ## run by DGL sparse API
        model.train()
        logits = model(params, batched_g.ndata["feat"])
        loss = loss_fcn(logits.squeeze(), batched_g.ndata["label"].float())
        loss.backward()
        print(model.MHA.q_proj.weight.grad.shape)
        print(model.MHA.k_proj.weight.grad.shape)
        print(model.MHA.v_proj.weight.grad.shape)
        model.MHA.q_proj.weight.grad
        v_grad = model.MHA.v_proj.weight.grad
        q_grad = model.MHA.q_proj.weight.grad
        k_grad = model.MHA.k_proj.weight.grad

        model.zero_grad()
        # print(model.MHA.q_proj.weight.grad)
        # print(model.MHA.k_proj.weight.grad)
        # print(model.MHA.v_proj.weight.grad)
        model.train()
        logits_fused = model(params, batched_g.ndata["feat"], fuse=True)
        loss = loss_fcn(logits_fused.squeeze(), batched_g.ndata["label"].float())
        print("check forward correct")
        check_correct(logits, logits_fused, params)
        loss.backward()
        ## Backward check
        print("check backward correct")
        # print(model.MHA.q_proj.weight.grad)
        # print(v_grad[0])
        # print(model.MHA.v_proj.weight.grad[0])
        print("V grad")
        check_correct(v_grad, model.MHA.v_proj.weight.grad, params)
        print("Q grad")
        check_correct(q_grad, model.MHA.q_proj.weight.grad, params)
        print("K grad")
        check_correct(k_grad, model.MHA.k_proj.weight.grad, params)

        break


def main(args):

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model
    dataset, train_fn, collate_fn = load_dataset_fn(args.dataset, args.data_dir)
    layer = SparseMHA_hyper(args.dim, args.heads)

    preprocess_func = preprocess_Hyper_fw_bw
    model = GTModel(layer, hidden_size=args.dim, outsize=1)
    model = model.to(dev)
    print("model", model)

    train_dataloader = GraphDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    fuse_flag = args.fused
    train(preprocess_func, model, train_dataloader, dev, fuse_flag)


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="DOTGAT")
    parser.add_argument("--fused", action="store_true")
    args = parser_argument(parser)
    main(args)
