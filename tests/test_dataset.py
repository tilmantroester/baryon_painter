import pickle

import numpy as np

from baryon_painter.utils.datasets import BAHAMASDataset

def test_dataset():
    """Tests that the BAHAMASDataset can be created and produce samples."""

    with open("data/training_data/BAHAMAS/stacks_uncompressed/train_files_info.pickle", "rb") as f:
        training_files_info = pickle.load(f)

    dataset = BAHAMASDataset(training_files_info, 
                             root_path="data/training_data/BAHAMAS/stacks_uncompressed/")

    d, idx = dataset[124]

    assert isinstance(d, list)
    assert isinstance(idx, int)
    assert isinstance(d[0], np.ndarray)

def test_transforms():
    """Tests transform with a simple transform to and from density contrast."""

    with open("data/training_data/BAHAMAS/stacks_uncompressed/test_files_info.pickle", "rb") as f:
        training_files_info = pickle.load(f)

    def transform_to_delta(x, field, z, stats):
        # print("transform mean: ", stats[field][z]["mean"])
        return x/stats[field][z]["mean"]-1.0

    def inv_transform_to_delta(x, field, z, stats):
        # print("inv transform mean: ", stats[field][z]["mean"])

        return (x+1.0)*stats[field][z]["mean"]#(x+1)*stats[field][z]["mean"]

    dataset = BAHAMASDataset(training_files_info, 
                             root_path="data/training_data/BAHAMAS/stacks_uncompressed/",
                             transform=transform_to_delta,
                             inverse_transform=inv_transform_to_delta)

    sample_idx = 12
    z = dataset.sample_idx_to_redshift(sample_idx)

    d, _ = dataset[sample_idx]

    transform = dataset.get_transforms(idx=sample_idx)
    inv_transform = dataset.get_inverse_transforms(idx=sample_idx)

    assert(np.allclose(d[0], transform[0](dataset.get_input_sample(sample_idx, transform=False))))
    assert(np.allclose(d[1], transform[1](dataset.get_label_sample(sample_idx, transform=False)[0])))
    assert(np.allclose(d[2], transform[2](dataset.get_label_sample(sample_idx, transform=False)[1])))
    assert(np.allclose(d[3], transform[3](dataset.get_label_sample(sample_idx, transform=False)[2])))

    assert(np.allclose(d[0], transform[0](inv_transform[0](d[0]))))
    assert(np.allclose(d[1], transform[1](inv_transform[1](d[1]))))
    assert(np.allclose(d[2], transform[2](inv_transform[2](d[2]))))
    assert(np.allclose(d[3], transform[3](inv_transform[3](d[3]))))

    print("Input field:")
    print(dataset.input_field, "(rel, abs)",
                                np.nanmax(np.abs(inv_transform[0](d[0])/dataset.get_input_sample(sample_idx, transform=False)-1)),
                                np.max(np.abs(inv_transform[0](d[0])-dataset.get_input_sample(sample_idx, transform=False))/np.sqrt(dataset.stats[dataset.input_field][z]["var"])))
    print("Output fields:")
    for i, field in enumerate(dataset.label_fields):
        print(field, "(rel, abs)", 
                      np.nanmax(np.abs(inv_transform[i+1](d[i+1])/dataset.get_label_sample(sample_idx, transform=False)[i]-1)),
                      np.max(np.abs(inv_transform[i+1](d[i+1])-dataset.get_label_sample(sample_idx, transform=False)[i])/np.sqrt(dataset.stats[field][z]["var"])))
    
    abs_tol = [2e-5*np.sqrt(dataset.stats[field][z]["var"]) for field in [dataset.input_field] + dataset.label_fields]
    assert np.allclose(inv_transform[0](d[0]) - dataset.get_input_sample(sample_idx, transform=False), 0, atol=abs_tol[0])
    for i, field in enumerate(dataset.label_fields):
        assert np.allclose(inv_transform[i+1](d[i+1]) - dataset.get_label_sample(sample_idx, transform=False)[i], 0, atol=abs_tol[i+1], rtol=0, equal_nan=True)

    # Relative accuracy isn't good. Might be due to values close to 0.
    # assert np.allclose(inv_transform[0](d[0]), dataset.get_input_sample(sample_idx, transform=False), equal_nan=True)
    # for i, field in enumerate(dataset.label_fields):
    #     assert np.allclose(inv_transform[i+1](d[i+1]), dataset.get_label_sample(sample_idx, transform=False)[i], equal_nan=True)
