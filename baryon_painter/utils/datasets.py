import os
import collections

import numpy as np

class BAHAMASDataset:
    """Dataset that deals with loading the BAHAMAS stacks.
    
    Arguments:
        files (list): List of dicts describing the data files. See below for a 
            description of the required entries.
        root_path (str, optional): Path where the data files will be looked for. 
            If not provided, the data files are expected to be in the current 
            directory or specified with absolute paths. (default None).
        redshifts (list, numpy.ndarray, optional): Redshifts that should be 
            included. If not provided, uses all redshifts in the files. 
            (default None).
        input_field (str, optional): Field to be used as input (default ``"dm"``).
        label_fields (list, optional): List of fields to be used as labels. If not 
            provided uses all fields in the files (except input_field). 
            (default None).
        n_tile (int, optional): Number of tiles per stack, where the total 
            number of tiles is n_tile^2. (default 4).
        transform (callable, optional): Transform to be applied to the samples. The 
            callable needs to have the signature ``f(x, field, z, **kwargs)``, 
            where ``x`` is the data to be transformed, ``field`` the field of 
            the data, and ``z`` the redshift. 
            (default ``lambda x, field, z, **kwargs: x``).
        inverse_transform (callable, optional): Inverse transform. The required 
            signature is the same as for the transform. 
            (default ``lambda x, field, z, **kwargs: (field, z, kwargs["mean"], kwargs["var"])``).
        verbose (bool, optional): Verbosity of the output (default False).
    """
    def __init__(self, files, root_path=None,
                 redshifts=None,
                 input_field="dm", label_fields=None, 
                 n_tile=4, 
                 transform=lambda x, field, z, **kwargs: x, 
                 inverse_transform=lambda x, field, z, **kwargs: (field, z, kwargs["mean"], kwargs["var"]), 
                 verbose=False):
        self.data = {}
        
        for f in files:
            if isinstance(f, dict):
                field = f["field"]
                z = f["z"]
                if field not in self.data:
                    self.data[field] = {}
                if z not in self.data[field]:
                    self.data[field][z] = {}

                fn100 = f["file_100"]
                fn150 = f["file_150"]
                if root_path is not None:
                    fn100 = os.path.join(root_path, fn100)
                    fn150 = os.path.join(root_path, fn150)
                    
                self.data[field][z]["100"] = np.load(fn100, mmap_mode="r")
                self.data[field][z]["150"] = np.load(fn150, mmap_mode="r")
                
                self.data[field][z]["mean_100"] = f["mean_100"]
                self.data[field][z]["mean_150"] = f["mean_150"]
                self.data[field][z]["var_100"] = f["var_100"]
                self.data[field][z]["var_150"] = f["var_150"]
                
                self.n_stack_100, self.n_grid, _ = self.data[field][z]["100"].shape
                self.n_stack_150, _, _ = self.data[field][z]["150"].shape
            else:
                raise ValueError("files entry is not a dict.")
        
        self.n_tile = n_tile
        self.tile_size = self.n_grid//self.n_tile
        self.n_sample = (self.n_stack_100*self.n_tile**2)*(self.n_stack_150*self.n_tile**2)
        
        self.fields = list(self.data.keys())
            
        self.input_field = input_field
        self.label_fields = [f for f in self.fields if f != self.input_field]
        if label_fields is not None:
            self.label_fields = label_fields
            
        self.redshifts = np.array(sorted(list(set([z for f in self.data.values() for z in f.keys()]))), ndmin=1)
        if redshifts is not None:
            self.redshifts = np.array(redshifts, ndmin=1)
            
        self.transform = transform
        self.inverse_transform = inverse_transform
        
    def create_inverse_transform(self, field, z, d, **stats):
        """Creates a callable for the inverse transform of the form f(x)."""

        return lambda x: self.inverse_transform(x, field, z, original=d, **stats)
        
    
    def get_stack(self, field, z, flat_idx):
        """Returns a stack for a given field, redshift, and index.
        
        Arguments:
            field (str): Field of the requested stack.
            z (float): Redshift of the requested stack.
            flat_idx (int): Index of the requested stack.
            
        Returns:
            (tuple): Tuple containing:
                (2d numpy.array): 250 Mpc/h equivalent stack.
                (dict): Dictionary with statistics of the stack. At this point only 
                    contains the mean and variance of all stacks in the dataset.
        """

        flat_idx = flat_idx%self.n_sample
        
        idx = np.unravel_index(flat_idx, dims=(self.n_stack_100, self.n_tile, self.n_tile, 
                                               self.n_stack_150, self.n_tile, self.n_tile))
        
        slice_idx_100 = idx[0]
        slice_idx_150 = idx[3]
        tile_idx_100 = slice(idx[1]*self.tile_size, (idx[1]+1)*self.tile_size), slice(idx[2]*self.tile_size, (idx[2]+1)*self.tile_size)
        tile_idx_150 = slice(idx[4]*self.tile_size, (idx[4]+1)*self.tile_size), slice(idx[5]*self.tile_size, (idx[5]+1)*self.tile_size)
        d_100 = self.data[field][z]["100"][slice_idx_100][tile_idx_100]
        d_150 = self.data[field][z]["150"][slice_idx_150][tile_idx_150]
        
        mean_100 = self.data[field][z]["mean_100"]
        mean_150 = self.data[field][z]["mean_150"]
        var_100 = self.data[field][z]["var_100"]
        var_150 = self.data[field][z]["var_150"]
        
        stats = {"mean" : mean_100+mean_150,
                 "var"  : var_100+var_150}
        
        return d_100+d_150, stats
    
    def sample_idx_to_redshift(self, idx):
        """Converts an index into the corresponding redshift."""

        redshift_idx = idx//self.n_sample
        z = self.redshifts[redshift_idx]
        return z
    
    def get_input_sample(self, idx, transform=True):
        """Get a sample for the input field.

        Arguments:
            idx (int): The index of the sample.
            transform (bool, optional): Transform the data. If True, returns the 
                inverse transform. (default True). 
        
        Returns:
            (tuple): Tuple containing:
                (2d numpy.array): Stack for the input field and index ``idx``.
                (callable): Inverse transform (only if ``transform == True``).
        """

        z = self.sample_idx_to_redshift(idx)

        d_input, input_stats = self.get_stack(self.input_field, z, idx)
        if not transform:
            return d_input
        else:
            input_inv_transform = self.create_inverse_transform(self.input_field, z, d_input, **input_stats)
            d_input = self.transform(d_input, self.input_field, z, **input_stats)

            return d_input, input_inv_transform

    def get_label_sample(self, idx, transform=True):
        """Get a sample for the label fields.

        Arguments:
            idx (int): The index of the sample.
            transform (bool, optional): Transform the data. If True, returns the 
                inverse transform. (default True). 
        
        Returns:
            (tuple): Tuple containing:
                (list of 2d numpy.arrays): Stacks for the label fields and index ``idx``. 
                (callable): Inverse transform (only if `transform == True``).
        """

        z = self.sample_idx_to_redshift(idx)
        
        d_labels = []
        label_inv_transforms = []
        for label_field in self.label_fields:
            d, stats = self.get_stack(label_field, z, idx)
            
            if transform:
                inv_transform = self.create_inverse_transform(label_field, z, d, **stats)
                label_inv_transforms.append(inv_transform)
                d = self.transform(d, label_field, z, **stats)
            
            d_labels.append(d)
            
        if not transform:
            return d_labels
        else:
            return d_labels, label_inv_transforms
    
    def __len__(self):
        """Return total number of samples.

        The total number of samples is given by 
        ``(n_stack_100*n_tile**2)*(n_stack_150*n_tile**2)*len(redshifts)``.
        """
        return self.n_sample*len(self.redshifts)
    
    def __getitem__(self, idx):
        if not isinstance(idx, collections.Iterable):
            d_input, input_inv_transform = self.get_input_sample(idx)
            d_label, label_inv_transforms = self.get_label_sample(idx)
            
            return [d_input]+d_label, [input_inv_transform]+label_inv_transforms     

