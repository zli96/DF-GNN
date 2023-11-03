"""
    File to load dataset based on user control from main file
"""
import os.path as osp

import ssl
import sys
import urllib
from typing import Optional

from dgl.data import (
    COCOSuperpixelsDataset,
    PeptidesFunctionalDataset,
    PeptidesStructuralDataset,
    VOCSuperpixelsDataset,
)

from dgl.data.utils import makedirs

from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.superpixels import SuperPixDataset


def download_url(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def LoadData(DATASET_NAME, data_dir):
    """
    This function is called in the main.py file
    returns:
    ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == "MNIST" or DATASET_NAME == "CIFAR10":
        return SuperPixDataset(DATASET_NAME, data_dir)

    # handling for (ZINC) molecule dataset
    if DATASET_NAME in ["ZINC", "ZINC-full", "AQSOL"]:
        return MoleculeDataset(DATASET_NAME, data_dir)

    # handling for SBM datasets
    SBM_DATASETS = ["SBM_CLUSTER", "SBM_PATTERN"]
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)

    # handling for LRGB datasets
    LRGB_DATASETS = ["PascalVOC-SP", "COCO-SP", "Peptides-func", "Peptides-struct"]
    if DATASET_NAME in LRGB_DATASETS:
        if DATASET_NAME == "PascalVOC-SP":
            return VOCSuperpixelsDataset(data_dir)
        elif DATASET_NAME == "COCO-SP":
            return COCOSuperpixelsDataset(data_dir)
        elif DATASET_NAME == "Peptides-func":
            return PeptidesFunctionalDataset(data_dir)
        elif DATASET_NAME == "Peptides-struct":
            return PeptidesStructuralDataset(data_dir)

    raise ValueError("Unknown dataset: {}".format(DATASET_NAME))
