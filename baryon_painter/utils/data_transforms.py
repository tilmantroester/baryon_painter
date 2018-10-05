import numpy as np

from scipy.ndimage import gaussian_filter

def transform_to_delta(x, field, z, **kwargs):
    return x/kwargs["mean"]-1

def inv_transform_to_delta(x, field, z, **kwargs):
    return (x+1)*kwargs["mean"]

from cosmotools.utils import rebin_2d

def create_split_scale_transform(n_scale=3, step_size=4, include_original=True):    
    def split_scale_transform(x, field, z, **kwargs):
        in_shape = np.array(x.shape)
        d_in = x.copy()
        if include_original:
            d_out = np.zeros((n_scale+1, *x.shape[-2:]), dtype=x.dtype)
            d_out[0] = x
        else:
            d_out = np.zeros((n_scale, *x.shape[-2:]), dtype=x.dtype)
        for i in range(n_scale-1, 0, -1):
            idx = i+1 if include_original else i
            # https://stackoverflow.com/a/32846903
            # d_out[i] = rebin_2d(d_in, in_shape//(step_size**i)).repeat(step_size**i, axis=0).repeat(step_size**i, axis=1)
            d_out[idx] = gaussian_filter(d_in, sigma=step_size**i/2, truncate=3.0)
            d_in -= d_out[idx]
        d_out[int(include_original)] = d_in
        return d_out
    
    return split_scale_transform
