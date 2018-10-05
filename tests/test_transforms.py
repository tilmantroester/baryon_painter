import numpy as np

from baryon_painter.utils.data_transforms import create_split_scale_transform

def test_split_scale_transform():
    n = 256
    m = np.random.randn(n,n)

    split_scale_transform = create_split_scale_transform(n_scale=3, step_size=2,
                                                         include_original=True)

    t = split_scale_transform(m, None, None)

    assert np.allclose(m, t[0])
    assert np.allclose(m, t[1:].sum(axis=0))

if __name__ == "__main__":
    test_split_scale_transform()