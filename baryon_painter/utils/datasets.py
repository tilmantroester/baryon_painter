import os
import collections

import numpy as np

import copy

def compile_transform(transform, stats={}, field=None, z=None):
    func = copy.deepcopy(transform)
    s = copy.deepcopy(stats)
    f = copy.deepcopy(field)
    z_ = copy.deepcopy(z)
    return lambda x, field=f, z=z_: func(x, field, z, s)

class BAHAMASDataset:
    """Dataset that deals with loading the BAHAMAS stacks.
    
    Arguments
    ---------
    files :list
        List of dicts describing the data files. See below for a description of 
        the required entries.
    root_path :str, optional
        Path where the data files will be looked for. If not provided, the data 
        files are expected to be in the current directory or specified with 
        absolute paths. (default None).
    redshifts : list, numpy.ndarray, optional
        Redshifts that should be included. If not provided, uses all redshifts 
        in the files. (default None).
    input_field : str, optional
        Field to be used as input (default ``"dm"``).
    label_fields : list, optional
        List of fields to be used as labels. If not provided uses all fields in 
        the files (except input_field). (default None).
    n_tile : int, optional
        Number of tiles per stack, where the total number of tiles is n_tile^2. 
        (default 4).
    transform : callable, optional
        Transform to be applied to the samples. The callable needs to have the 
        signature ``f(x, field, z, stats)``, where ``x`` is the data to be 
        transformed, ``field`` the field of the data, ``z`` the redshift, and
        ``stats`` is a dict with statistics of the data set. 
        (default ``lambda x, field, z, stats: x``).
    inverse_transform : callable, optional
        Inverse transform. The required signature is the same as for the 
        transform. 
        (default ``lambda x, field, z, stats: x``).
    scale_to_SLICS : bool, optional
        Scale dark matter to match to SLICS delta-planes.
        (default True).
    subtract_minimum : bool, optional
        Subtract minimum of dark matter field to match to SLICS density-planes.
        (default True).
    verbose : bool, optional
        Verbosity of the output (default False).
    """
    def __init__(self, files, root_path=None,
                 redshifts=[],
                 input_field="dm", label_fields=[], 
                 n_tile=4,
                 L=400,
                 transform=lambda x, field, z, stats: x, 
                 inverse_transform=lambda x, field, z, stats: x,
                 n_feature_per_field=1,
                 scale_to_SLICS=True,
                 subtract_minimum=False,
                 mmap_mode="r",
                 verbose=False):
        self.data = {}
        
        self.fields = []
        self.redshifts = []
        
        # Check which fields and redshifts are available
        for f in files:
            if isinstance(f, dict):
                self.fields.append(f["field"])
                self.redshifts.append(f["z"])
            else:
                raise ValueError("files entry is not a dict.")
        
        # Get unique values for fields and redshifts while preserving the order
        # they appeared at in `files`.
        # Since Python 3.7, this can be done with just a dict but we're not that
        # aggressive yet.
        self.fields = list(collections.OrderedDict.fromkeys(self.fields))
        self.redshifts = list(collections.OrderedDict.fromkeys(self.redshifts))

        self.input_field = input_field

        if label_fields != []:
            self.label_fields = label_fields
            # Select the intersection of the available fields and requested fields.
            if input_field in self.fields and all([f in self.fields for f in label_fields]):
                self.fields = [input_field] + label_fields
            else:
                missing = set([input_field] + label_fields) - set(self.fields)
                raise ValueError(f"The requested fields are not in the file list: field(s) {missing} is missing.")
        else:
            self.label_fields = [f for f in self.fields if f != self.input_field]
        
        if redshifts != []:
            # Select the intersection of the available redshifts and requested redshifts.
            if all([z in self.redshifts for z in redshifts]):
                self.redshifts = redshifts
            else:
                missing = set(redshifts) - set(self.redshifts)
                raise ValueError(f"The requested redshifts are not in the file list: redshift(s) {missing} is missing.")
        else:
            self.redshifts = np.array(sorted(list(self.redshifts)))
              
        # Load the files now
        for f in files:
            field = f["field"]
            z = f["z"]
            if field not in self.fields or z not in self.redshifts:
                # Don't load fields that are not requested
                continue
                
            if field not in self.data:
                self.data[field] = {}
            if z not in self.data[field]:
                self.data[field][z] = {}
                    
            fn100 = f["file_100"]
            fn150 = f["file_150"]
            if root_path is not None:
                fn100 = os.path.join(root_path, fn100)
                fn150 = os.path.join(root_path, fn150)

            self.data[field][z]["100"] = np.load(fn100, mmap_mode=mmap_mode)
            self.data[field][z]["150"] = np.load(fn150, mmap_mode=mmap_mode)

            self.data[field][z]["mean_100"] = f["mean_100"]
            self.data[field][z]["mean_150"] = f["mean_150"]
            self.data[field][z]["var_100"] = f["var_100"]
            self.data[field][z]["var_150"] = f["var_150"]

            self.n_stack_100, self.n_grid, _ = self.data[field][z]["100"].shape
            self.n_stack_150, _, _ = self.data[field][z]["150"].shape
        
        self.n_tile = n_tile
        self.tile_size = self.n_grid//self.n_tile
        self.n_sample = (self.n_stack_100*self.n_tile**2)*(self.n_stack_150*self.n_tile**2)

        self.L = L
        self.tile_L = self.L/self.n_tile
                                
        self.transform_func = transform
        self.inverse_transform_func = inverse_transform

        self.n_feature_per_field = n_feature_per_field
        
        self.scale_to_SLICS = scale_to_SLICS
        self.subtract_minimum = subtract_minimum
        
        self.stats = collections.OrderedDict()
        for field in self.fields:
            self.stats[field] = collections.OrderedDict()
            for z in self.redshifts:
                self.stats[field][z] = self.get_stack_stats(field, z)
                
        self.transform = compile_transform(transform, self.stats)
        self.inverse_transform = compile_transform(inverse_transform, self.stats)

        

    def create_transform(self, field, z):
        """Creates a callable for the transform of the form f(x)."""

        return compile_transform(self.transform_func, self.stats, field, z)
    
    def create_inverse_transform(self, field, z):
        """Creates a callable for the inverse transform of the form f(x)."""

        return compile_transform(self.inverse_transform_func, self.stats, field, z)
        
    def get_transforms(self, idx=None, z=None):
        """Get the transforms for a stack.

        Arguments
        ---------
        idx : int, optional
            Index of the stack.
        z : float, optional
            Redshift of the stack.
            
        Either ``idx`` or ``z`` have to be specified.
        
        Returns
        -------
        transforms : list
            List of the transforms for the input and label fields.
        """
        if idx is None and z is None:
            raise ValueError("Either idx or z have to be specified.")
            
        if z is None:
            z = self.sample_idx_to_redshift(idx)

        transforms = []
        for field in [self.input_field]+self.label_fields:
            transforms.append(self.create_transform(field, z))

        return transforms
    
    def get_inverse_transforms(self, idx=None, z=None):
        """Get the inverse transforms for a stack.

        Arguments
        ---------
        idx : int, optional
            Index of the stack.
        z : float, optional
            Redshift of the stack.
            
        Either ``idx`` or ``z`` have to be specified.
            

        Returns
        -------
        inv_transforms : list
            List of the inverse transforms for the input and label fields.
        """
        if idx is None and z is None:
            raise ValueError("Either idx or z have to be specified.")
            
        if z is None:
            z = self.sample_idx_to_redshift(idx)

        inv_transforms = []
        for field in [self.input_field]+self.label_fields:
            inv_transforms.append(self.create_inverse_transform(field, z))

        return inv_transforms

    def get_stack_stats(self, field, z):
        """Returns stack stats for a given field and redshift.
        
        Arguments
        ---------
        field : str
            Field of the requested stack.
        z : float
            Redshift of the requested stack.
            
        Returns
        -------
        stats : dict
            Dictionary with statistics of the stack. At this point only contains 
            the mean and variance of all stacks in the dataset.
        """
        
        mean_100 = self.data[field][z]["mean_100"]
        mean_150 = self.data[field][z]["mean_150"]
        var_100 = self.data[field][z]["var_100"]
        var_150 = self.data[field][z]["var_150"]
        
        stats = {"mean" : mean_100+mean_150,
                 "var"  : var_100+var_150}
        
        if field == self.input_field and self.scale_to_SLICS:
            stats["mean"] *= 1/(self.n_grid/8*5)*0.2793/(0.2793-0.0463)
            stats["var"] *= (1/(self.n_grid/8*5)*0.2793/(0.2793-0.0463))**2
        return stats

    def get_stack(self, field, z, flat_idx):
        """Returns a stack for a given field, redshift, and index.
        
        Arguments
        ---------
        field : str
            Field of the requested stack.
        z : float
            Redshift of the requested stack.
        flat_idx : int
            Index of the requested stack.
            
        Returns
        -------
        stack : 2d numpy.array
            250 Mpc/h equivalent stack.
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
                
        return d_100+d_150
    
    def sample_idx_to_redshift(self, idx):
        """Converts an index into the corresponding redshift."""

        redshift_idx = idx//self.n_sample
        z = self.redshifts[redshift_idx]
        return z
    
    def get_input_sample(self, idx, transform=True):
        """Get a sample for the input field.

        Arguments
        ---------
        idx : int
            The index of the sample.
        transform :bool, optional
            Transform the data. If True, returns the  inverse transform. 
            (default True). 
        
        Returns
        -------
        output : 2d numpy.array
            Stack for the input field and index ``idx``.
        """

        z = self.sample_idx_to_redshift(idx)

        d_input = self.get_stack(self.input_field, z, idx)
        if self.scale_to_SLICS:
            d_input = 1/(self.n_grid/8*5)*0.2793/(0.2793-0.0463)*(d_input-d_input.mean())
        if self.subtract_minimum:
            d_input -= d_input.min()
        if transform:
            d_input = self.transform(d_input, self.input_field, z)
        return d_input

    def get_label_sample(self, idx, transform=True):
        """Get a sample for the label fields.

        Arguments
        ---------
        idx : int
            The index of the sample.
        transform : bool, optional
            Transform the data. If True, returns the inverse transform. 
            (default True). 
        
        Returns
        -------
        output : list
            List of stacks for the input field and index ``idx``.
        """

        z = self.sample_idx_to_redshift(idx)
        
        d_labels = []
        for label_field in self.label_fields:
            d = self.get_stack(label_field, z, idx)
            if transform:
                d = self.transform(d, label_field, z)
            d_labels.append(d)
            
        return d_labels
    
    def __len__(self):
        """Return total number of samples.

        The total number of samples is given by 
        ``(n_stack_100*n_tile**2)*(n_stack_150*n_tile**2)*len(redshifts)``.
        """
        return self.n_sample*len(self.redshifts)
    
    def __getitem__(self, idx):
        """Get a sample.

        Arguments
        ---------
        idx : int
            Index of the sample.

        Returns
        -------
        output : list
            List of sample fields, with order ``input_field, label_fields``.
        idx : int
            Index of the requested sample. This can be used to access the
            inverse transforms.
        z : float
            Redshift of the requested sample.
        """
        if not isinstance(idx, collections.abc.Iterable):
            d_input = self.get_input_sample(idx)
            d_label = self.get_label_sample(idx)
            
            return [d_input]+d_label, idx, self.sample_idx_to_redshift(idx)
        else:
            raise NotImplementedError("Only int indicies are supported for now.")

