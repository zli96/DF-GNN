import os, sys
import os.path as osp
import pickle
import shutil

import dgl
import torch
from dgl.data.utils import Subset
from dgl.dataloading import GraphDataLoader
from ogb.utils.url import extract_zip, makedirs
from tqdm import tqdm

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)

import data.dataset


class VOCSuperpixels_DGL(object):
    r"""The VOCSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.

    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels.
    - In VOCSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node

    - Based on the SBD annotations from 11355 images taken from the PASCAL VOC 2011 dataset. Original
    source `here<https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal>`_.

    num_classes = 21
    ignore_label = 255

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train,
    20=tv/monitor

    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For VOCSuperpixels, we maintain train, val and test splits where the train set is AS IS. The original
    val split of the image dataset is used to divide into new val and new test split that is eventually used
    in VOCSuperpixels. The policy for this val/test splitting is below.
    - Split total number of val graphs into 2 sets (val, test) with 50:50 using a stratified split proportionate
    to original distribution of data with respect to a meta label.
    - Each image is meta-labeled by majority voting of non-background grouth truth node labels. Then new val
    and new test is created with stratified sampling based on these meta-labels. This is done for preserving
    same distribution of node labels in both new val and new test
    - Therefore, the final train, val and test splits are correspondingly original train (8498), new val (1428)
    and new test (1429) splits.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e.
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = {
        10: {
            "edge_wt_only_coord": "https://www.dropbox.com/s/rk6pfnuh7tq3t37/voc_superpixels_edge_wt_only_coord.zip?dl=1",
            "edge_wt_coord_feat": "https://www.dropbox.com/s/2a53nmfp6llqg8y/voc_superpixels_edge_wt_coord_feat.zip?dl=1",
            "edge_wt_region_boundary": "https://www.dropbox.com/s/6pfz2mccfbkj7r3/voc_superpixels_edge_wt_region_boundary.zip?dl=1",
        },
        30: {
            "edge_wt_only_coord": "https://www.dropbox.com/s/toqulkdpb1jrswk/voc_superpixels_edge_wt_only_coord.zip?dl=1",
            "edge_wt_coord_feat": "https://www.dropbox.com/s/xywki8ysj63584d/voc_superpixels_edge_wt_coord_feat.zip?dl=1",
            "edge_wt_region_boundary": "https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1",
        },
    }

    def __init__(
        self,
        root="data",
        name="edge_wt_region_boundary",
        slic_compactness=30,
        split="train",
    ):
        self.root = osp.join(root, "voc_superpixels")
        self.name = name
        self.slic_compactness = slic_compactness
        assert split in ["train", "val", "test"]
        assert name in [
            "edge_wt_only_coord",
            "edge_wt_coord_feat",
            "edge_wt_region_boundary",
        ]
        assert slic_compactness in [10, 30]
        self.split = split
        self.preprocesspath = osp.join(self.processed_dir, f"{split}.pkl")
        self.process()

    @property
    def raw_file_names(self):
        return ["train.pickle", "val.pickle", "test.pickle"]

    @property
    def raw_dir(self):
        return osp.join(
            self.root,
            "slic_compactness_" + str(self.slic_compactness),
            self.name,
            "raw",
        )

    @property
    def processed_dir(self):
        return osp.join(
            self.root,
            "slic_compactness_" + str(self.slic_compactness),
            self.name,
            "processed",
        )

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 21

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.graphs)

    def download(self):
        makedirs(self.raw_dir)
        shutil.rmtree(self.raw_dir)
        path = data.dataset.download_url(
            self.url[self.slic_compactness][self.name], self.root
        )
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, "voc_superpixels_" + self.name), self.raw_dir)
        os.unlink(path)

    def process(self):
        if osp.exists(self.preprocesspath):
            # if pre-processed file already exists
            with open(self.preprocesspath, "rb") as f:
                f = pickle.load(f)
                self.graphs = f
        else:
            if not osp.exists(osp.join(self.raw_dir, f"{self.split}.pickle")):
                # if the raw file does not exist, then download it.
                self.download()

            makedirs(self.processed_dir)
            for split in ["train", "val", "test"]:
                with open(osp.join(self.raw_dir, f"{split}.pickle"), "rb") as f:
                    graphs = pickle.load(f)

                indices = range(len(graphs))

                pbar = tqdm(total=len(indices))
                pbar.set_description(f"Processing {split} dataset")

                self.graphs = []
                for idx in indices:
                    graph = graphs[idx]

                    """
                    Each `graph` is a tuple (x, edge_attr, edge_index, y)
                        Shape of x : [num_nodes, 14]
                        Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                        Shape of edge_index : [2, num_edges]
                        Shape of y : [num_nodes]
                    """
                    dgl_graph = dgl.graph(
                        (graph[2][0], graph[2][1]),
                        num_nodes=len(graph[3]),
                    )
                    dgl_graph.ndata["feat"] = graph[0].to(torch.float)
                    dgl_graph.edata["feat"] = graph[1].to(torch.float)
                    dgl_graph.ndata["label"] = torch.LongTensor(graph[3])
                    self.graphs.append(dgl_graph)

                    pbar.update(1)

                pbar.close()

                print("Saving...")
                print("# of graphs", len(self.graphs))
                with open(osp.join(self.processed_dir, f"{split}.pkl"), "wb") as f:
                    pickle.dump(self.graphs, f)

    def __getitem__(self, idx):
        r"""Get the idx^th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features, node labels and edge features.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
            - ``edata['feat']``: edge features
        """

        if isinstance(idx, int):
            return self.graphs[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            "Only integers and long are valid "
            "indices (got {}).".format(type(idx).__name__)
        )


if __name__ == "__main__":
    dataset = VOCSuperpixels_DGL()
    print(dataset)
    print("# of classes for each node", dataset.num_classes)
    print(dataset[0])
    train_dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=False)
    for i, batched_g in enumerate(train_dataloader):
        print("batched graph", batched_g)
        assert batched_g.num_nodes() == batched_g.ndata["label"].shape[0]
        break
