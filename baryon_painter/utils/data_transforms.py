import numpy as np

from scipy.ndimage import gaussian_filter

from cosmotools.utils import rebin_2d

def transform_to_delta(x, field, z, stats):
    return x/stats[field][z]["mean"]-1

def inv_transform_to_delta(x, field, z, stats):
    return (x+1)*stats[field][z]["mean"]
    

def create_split_scale_transform(n_scale=3, step_size=4, include_original=True, truncate=3.0):    
    def split_scale_transform(x, field, z, stats):
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
            d_out[idx] = gaussian_filter(d_in, sigma=step_size**i/2, truncate=truncate)
            d_in -= d_out[idx]
        d_out[int(include_original)] = d_in
        return d_out
    
    def inv_split_scale_transform(x, field, z, stats):
        if include_original:
            if x.shape[0] != n_scale+1:
                raise RuntimeError(f"Invalid shape of input. Expected x.shape[0] == {n_scale+1} but got {x.shape[0]}.")
            return x[0]
        else:
            if x.shape[0] != n_scale:
                raise RuntimeError(f"Invalid shape of input. Expected x.shape[0] == {n_scale} but got {x.shape[0]}.")
            return x.sum(axis=0)
        
    return split_scale_transform, inv_split_scale_transform

def chain_transformations(transformations):
    def transform(x, field, z, stats):
        for t in transformations:
            x = t(x, field, z, stats)
        return x
    return transform
    
def create_range_compress_transforms(k_values, modes={}, eps=1e-3, sqrt_of_mean=False):
    def interpolate_z(stats, z):
        """Interpolate statisitcs dict to redshift z."""
        z_list = list(stats.keys())
        idx = np.searchsorted(z_list, z, side="right")
        if idx >= len(z_list):
            return stats[z_list[-1]]
        elif idx <= 0:
            return stats[z_list[0]]
        w = (z - z_list[idx-1])/(z_list[idx]-z_list[idx-1])
        stats_names = stats[z_list[0]].keys()
        intp_stats = {s : w*stats[z_list[idx]][s] + (1-w)*stats[z_list[idx-1]][s] for s in stats_names}

        return intp_stats

    def transform(x, field, z, stats):
        k = k_values[field]
        mode = modes[field]
        mean = interpolate_z(stats[field], z)["mean"]
        if sqrt_of_mean: mean = np.sqrt(mean)
        std = np.sqrt(interpolate_z(stats[field], z)["var"])
        if mode.lower() == "log":
            return np.where(x > 0, np.log(x/std+eps)/k, np.log(eps)/k)
        elif mode.lower() == "shift-log":
#             return np.where(x >= 0, np.log(x/std+1)/k, 0)
            return np.log(x/std+1)/k
        elif mode.lower() == "shift-log-2p":
            return np.log(x/std+k[0])/k[1]
        elif mode.lower() == "log-tanh":
            return np.where(x > 0, np.tanh(np.log(x/std+eps)/k), -1)
        elif mode.lower() == "x/(1+x)":
            return x/(x+std)*k[0]-k[1]
        elif mode.lower() == "1/x":
            return np.where(x/(std*mean*k)>-1, 2/(x/(std*mean*k)+1) - 1.001, -1)
        else:
             raise ValueError(f"Mode '{mode}' not supported.")               
    
    def inv_transform(x, field, z, stats):
        k = k_values[field]
        mode = modes[field]
        mean = interpolate_z(stats[field], z)["mean"]
        if sqrt_of_mean: mean = np.sqrt(mean)
        std = np.sqrt(interpolate_z(stats[field], z)["var"])
        if mode.lower() == "log":
            return np.where(x > np.log(eps)/k, (np.exp(x*k)-eps)*std, 0)
        elif mode.lower() == "shift-log":
#             return np.where(x >= 0, (np.exp(x*k)-1)*std, 0)
            return (np.exp(x*k)-1)*std
        elif mode.lower() == "shift-log-2p":
            return (np.exp(x*k[1])-k[0])*std
        elif mode.lower() == "log-tanh":
            return np.where(x > -1, (np.exp(np.arctanh(x)*k)-eps)*std, 0)
        elif mode.lower() == "x/(1+x)":
            return std/(k[0]/(x+k[1])-1)
        elif mode.lower() == "1/x":
            return np.where(x >= -1 , (2/(x+1.001) - 1)*std*mean*k, 0)
        else:
             raise ValueError(f"Mode '{mode}' not supported.")   
    
    return transform, inv_transform 

def atleast_3d(x, field, z, stats):
    if x.ndim == 2:
        return x.reshape(1, *x.shape)
    else:
        return x

def squeeze(x, field, z, stats):
    return x.squeeze()