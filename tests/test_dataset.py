import pickle

import numpy as np

from baryon_painter.utils.datasets import BAHAMASDataset

def test_dataset():
    """Tests that the BAHAMASDataset can be created and produce samples."""

    with open("data/training_data/BAHAMAS/stacks_uncompressed/train_files_info.pickle", "rb") as f:
        training_files_info = pickle.load(f)

    dataset = BAHAMASDataset(training_files_info, 
                             root_path="data/training_data/BAHAMAS/stacks_uncompressed/")

    d, inv_transforms = dataset[124]

    assert isinstance(d, list)
    assert isinstance(inv_transforms, list)
    assert isinstance(d[0], np.ndarray)
    assert callable(inv_transforms[0])

def test_transforms():
    """Tests transform with a simple transform to and from density contrast."""

    with open("data/training_data/BAHAMAS/stacks_uncompressed/train_files_info.pickle", "rb") as f:
        training_files_info = pickle.load(f)

    def transform_to_delta(x, field, z, **kwargs):
        return x/kwargs["mean"]-1

    def inv_transform_to_delta(x, field, z, **kwargs):
        return (x+1)*kwargs["mean"]

    dataset = BAHAMASDataset(training_files_info, 
                             root_path="data/training_data/BAHAMAS/stacks_uncompressed/",
                             transform=transform_to_delta,
                             inverse_transform=inv_transform_to_delta)

    sample_idx = 124

    d, inv_transforms = dataset[sample_idx]

    assert np.allclose(inv_transforms[0](d[0]), dataset.get_input_sample(sample_idx, transform=False))
    assert np.allclose(inv_transforms[1](d[1]), dataset.get_label_sample(sample_idx, transform=False)[0])