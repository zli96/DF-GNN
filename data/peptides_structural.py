import hashlib
import os, sys
import os.path as osp
import pickle
import shutil

import dgl

import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from tqdm import tqdm

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)

import data.dataset


class PeptidesStructuralDataset_DGL(object):
    def __init__(self, root="data", smiles2graph=smiles2graph):
        """
        DGL dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.

        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "peptides-structural")
        self.raw_dir = osp.join(self.folder, "raw")

        self.url = "https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1"
        self.version = (
            "9786061a34298a0684150f2e4ff13f47"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        self.process()

    @property
    def raw_file_names(self):
        return "peptide_structure_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = data.dataset.download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.folder, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = data.dataset.download_url(
                self.url_stratified_split, self.folder
            )
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        processed_dir = osp.join(self.folder, "processed")
        pre_processed_file_path = osp.join(processed_dir, "dgl_data_processed")

        if osp.exists(pre_processed_file_path):
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict["labels"]
        else:
            # if pre-processed file does not exist
            if not osp.exists(
                osp.join(self.raw_dir, "peptide_structure_dataset.csv.gz")
            ):
                # if the raw file does not exist, then download it.
                self.download()

            data_df = pd.read_csv(
                osp.join(self.raw_dir, "peptide_structure_dataset.csv.gz")
            )
            smiles_list = data_df["smiles"]
            target_names = [
                "Inertia_mass_a",
                "Inertia_mass_b",
                "Inertia_mass_c",
                "Inertia_valence_a",
                "Inertia_valence_b",
                "Inertia_valence_c",
                "length_a",
                "length_b",
                "length_c",
                "Spherocity",
                "Plane_best_fit",
            ]
            # Normalize to zero mean and unit standard deviation.
            data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            )

            print("Converting SMILES strings into graphs...")
            self.graphs = []
            self.labels = []
            for i in tqdm(range(len(smiles_list))):

                smiles = smiles_list[i]
                y = data_df.iloc[i][target_names]
                graph = self.smiles2graph(smiles)

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]
                dgl_graph = dgl.graph(
                    (graph["edge_index"][0], graph["edge_index"][1]),
                    num_nodes=graph["num_nodes"],
                )
                dgl_graph.edata["feat"] = torch.from_numpy(graph["edge_feat"]).to(
                    torch.int64
                )
                dgl_graph.ndata["feat"] = torch.from_numpy(graph["node_feat"]).to(
                    torch.int64
                )

                self.graphs.append(dgl_graph)
                self.labels.append(y)

            self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # double-check prediction target
            split_dict = self.get_idx_split()
            assert all(
                [not any(torch.isnan(self.labels[i])) for i in split_dict["train"]]
            )
            assert all(
                [not any(torch.isnan(self.labels[i])) for i in split_dict["val"]]
            )
            assert all(
                [not any(torch.isnan(self.labels[i])) for i in split_dict["test"]]
            )

            print("Saving...")
            save_graphs(
                pre_processed_file_path, self.graphs, labels={"labels": self.labels}
            )

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(
            self.folder, "splits_random_stratified_peptide_structure.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            "Only integers and long are valid "
            "indices (got {}).".format(type(idx).__name__)
        )


# Collate function for ordinary graph classification
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels


if __name__ == "__main__":
    dataset = PeptidesStructuralDataset_DGL()
    print(dataset)
    print(dataset[100])
    split_dict = dataset.get_idx_split()
    print(split_dict)
    print(dataset[split_dict["train"]])
    print(collate_dgl([dataset[0], dataset[1], dataset[2]]))
