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
    data : dict, optional
        Dictionary with the raw data and meta information. (default None).
    files :list, optional
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
    L : float, optional
        Physical size of the stacks. (default 400).
    n_stack : int, optional
        Number of stacks to use. If ``None``, uses all stacks in the files.
    stack_offset : int, optional
        Offset in the stacks to use. To separate a validation set, set ``stack_offset``
        to the ``n_stack`` used for the training set to avoid overlap. (default 0).
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
    tile_permutations : bool, optional
        Apply rotations and flips to the tiles to increase the number of samples
        by a factor of 16. (default False).
    scale_to_SLICS : bool, optional
        Scale dark matter to match to SLICS delta-planes.
        (default True).
    subtract_minimum : bool, optional
        Subtract minimum of dark matter field to match to SLICS density-planes.
        (default True).
    mmap_mode : string, optional
        Memory map mode that is used to load the files. Gets passed to numpy.load.
        (default ``"r"``).
    verbose : bool, optional
        Verbosity of the output (default False).
    """
    def __init__(self, data=None, files=None, root_path=None,
                 redshifts=[],
                 input_field="dm", label_fields=[], 
                 n_tile=4,
                 L=400,
                 n_stack=None, stack_offset=0,
                 transform=lambda x, field, z, stats: x, 
                 inverse_transform=lambda x, field, z, stats: x,
                 n_feature_per_field=1,
                 tile_permutations=False,
                 scale_to_SLICS=True,
                 subtract_minimum=False,
                 mmap_mode="r",
                 verbose=False):
        
        self.fields = []
        self.redshifts = []
        
        if data is not None:
            # If data structure is provided, use that to avoid duplicating memory
            self.data = data
            self.fields = list(self.data.keys())
            self.redshifts = list(self.data[self.fields[0]].keys())
        elif files is not None:
            # Check which fields and redshifts are available
            self.data = {}
            for f in files:
                if isinstance(f, dict):
                    self.fields.append(f["field"])
                    self.redshifts.append(f["z"])
                else:
                    raise ValueError("files entry is not a dict.")
        else:
            raise ValueError("Either data or files need to be provided.")
        
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
            self.redshifts = sorted(list(self.redshifts))
            
        if files is not None:
            # Load data from files
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
        

        self.n_stack_100, self.n_grid, _ = self.data[self.fields[0]][self.redshifts[0]]["100"].shape
        self.n_stack_150, _, _ = self.data[self.fields[0]][self.redshifts[0]]["150"].shape
        
        if n_stack is None:
            self.n_stack = min(self.n_stack_100, self.n_stack_150)
        else:
            self.n_stack = n_stack
        self.stack_offset = stack_offset
        
        if min(self.n_stack_100, self.n_stack_150) < self.stack_offset + self.n_stack:
            raise ValueError(f"Highest stack exceeds number of available stacks.")
        
        self.n_tile_permutation = 8 if tile_permutations else 1
        self.n_tile = n_tile
        self.tile_size = self.n_grid//self.n_tile
        self.n_total_sample = (self.n_stack_100*self.n_tile**2*self.n_tile_permutation)*(self.n_stack_150*self.n_tile**2*self.n_tile_permutation)
        self.n_sample = self.n_stack**2*self.n_tile**4*self.n_tile_permutation**2

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
            
        # Remove redshift offset
        no_z_idx = flat_idx%self.n_sample

        # Remove tile permutation offset
        no_z_no_perm_idx = no_z_idx%self.n_tile_permutation**2
        
#         print(f"Getting stack for field {field}, z {z}, idx {flat_idx}")
        idx = np.unravel_index(no_z_no_perm_idx, dims=(self.n_stack, self.n_tile, self.n_tile, 
                                                       self.n_stack, self.n_tile, self.n_tile))
        
        slice_idx_100 = idx[0] + self.stack_offset
        slice_idx_150 = idx[3] + self.stack_offset
        tile_idx_100 = slice(idx[1]*self.tile_size, (idx[1]+1)*self.tile_size), slice(idx[2]*self.tile_size, (idx[2]+1)*self.tile_size)
        tile_idx_150 = slice(idx[4]*self.tile_size, (idx[4]+1)*self.tile_size), slice(idx[5]*self.tile_size, (idx[5]+1)*self.tile_size)
        d_100 = self.data[field][z]["100"][slice_idx_100][tile_idx_100]
        d_150 = self.data[field][z]["150"][slice_idx_150][tile_idx_150]
        
        permutation_idx = self.sample_idx_to_tile_permutation(flat_idx)
        d_100 = self.apply_tile_permutation(d_100, permutation_idx)
        d_150 = self.apply_tile_permutation(d_100, permutation_idx)

        return d_100+d_150
    
    def apply_tile_permutation(self, tile, permutation_idx):
        """Apply rotations and flips to the tile."""

        rot_idx = permutation_idx//4
        flip_idx = permutation_idx%4
        if rot_idx > 0:
            tile = np.rot90(tile, k=rot_idx)
        if flip_idx == 1:
            tile = tile[:,::-1]
        elif flip_idx == 2:
            tile = tile[::-1]
        elif flip_idx == 2:
            tile = tile[::-1,::-1]
        return tile
        
    def sample_idx_to_redshift(self, idx):
        """Converts an index into the corresponding redshift."""

        redshift_idx = idx//self.n_sample
        z = self.redshifts[redshift_idx]
        return z

    def sample_idx_to_tile_permutation(self, idx):
        """Converts an index into the corresponding tile permutation."""

        sample_idx = idx%self.n_sample
        permutation_idx = idx//(self.n_sample//self.n_tile_permutation**2)
        return permutation_idx
    
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
            d_input = 1/(self.n_grid/8*5)*0.2793/(0.2793-0.0463)*d_input
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
    
    def get_batch(self, size=1, z=None, idx=None):
        """Get a batch from the data set by random sampling.
        
        Arguments
        ---------
        size : int, optional
            Number of samples to return. (default 1).
        z : float, optional
            Redshift of the requested samples. If ``None`` samples from all redshifts. (default ``None``).
        idx : numpy.array, optional
            Array of indicies of the requested samples. (default ``None``).
            
        Returns
        -------
        samples : numpy.array
            Array of shape (1+F_label, N, C, H, W), where N is the size of the batch (``size``), F_label is the number label fields, C the number of features, and H,W is the size of the tile in pixel.
        idx : numpy.array
            Array with the indicies of the samples.
        z : numpy.array
            Array with the redshifts of the samples.
        """
        
        if idx is None:
            idx = np.random.choice(self.n_sample, size=size, replace=False)
            if z is None:
                idx *= len(self.redshifts)
                z = [self.sample_idx_to_redshift(i) for i in idx]
            else:
                idx_offset = self.redshifts.index(z)*self.n_sample
                idx += idx_offset
                z = [z]*size
        else:
            z = [self.sample_idx_to_redshift(i) for i in idx]
            
        samples = []
        for i in idx:
            s, _, _ = self[i]
            samples.append(s)
        
        return np.array(samples).swapaxes(0,1), idx, np.array(z)
        
    def __len__(self):
        """Return total number of samples.

        The total number of samples is given by 
        ``n_stack_100**2**n_tile**4*len(redshifts)``.
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

