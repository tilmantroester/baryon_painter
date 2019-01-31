import torch

class Flatten(torch.nn.Module):
    """Flattens tensor."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
class UnFlatten(torch.nn.Module):
    """Unflattens tensor."""
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, x):
        x = x.view(x.size(0), *self.output_dim)
        return x
        
class ResidualBlock(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.res_block = build_sequential(architecture[0])
        if architecture[1][0].lower() == "relu":
            self.activation = torch.nn.ReLU(inplace=True)
        elif architecture[1][0].lower() == "leaky relu":
            self.activation = torch.nn.LeakyReLU(architecture[1][1], inplace=True)
        elif architecture[1][0] is None:
            self.activation = lambda x: x
        else:
            raise NotImplementedError("Layer {} not supported yet!".format(architecture[1][0]))
            
    def forward(self, x):
        x = torch.add(self.res_block(x), x)
        x = self.activation(x)
        return x

def conv_block(in_channel, out_channel, type="conv", scale=1, kernel=3, bias=False, batchnorm=True, activation="relu", relu_slope=0.2):
    if scale == 1:
        if kernel % 2 != 1:
            raise ValueError("Kernel with scale=1 should be odd.")
        padding = (kernel-1)//2
        kps_params = {"kernel_size" : kernel, "padding" : padding, "stride" : 1}
    elif scale == 2:
        kps_params = {"kernel_size" : 4, "padding" : 1, "stride" : 2}
    elif scale == 4:
        kps_params = {"kernel_size" : 8, "padding" : 2, "stride" : 4}
    else:
        raise NotImplementedError("Scaling {} not supported yet!".format(scale))
    
    architecture = [(type,    {"in_channels"  : in_channel, 
                               "out_channels" : out_channel,
                               **kps_params,
                               "bias"         : bias}),]
    if batchnorm:
        architecture += [("batchnorm", {"num_features" : out_channel}),]
    
    if activation is None or activation.lower() == "none":
        pass
    elif activation.lower() == "relu":
        architecture += [("ReLU",),]
    elif activation.lower() == "leaky relu":
        architecture += [("Leaky ReLU", relu_slope),]
    elif activation.lower() == "prelu":
        architecture += [("prelu",),]
    elif activation.lower() == "tanh":
        architecture += [("tanh",),]
    elif activation.lower() == "sigmoid":
        architecture += [("sigmoid",),]
    elif activation.lower() == "softplus":
        architecture += [("softplus",),]
    else:
        raise NotImplementedError("Activation {} not supported yet!".format(activation))
        
    return architecture

def res_block(n_channel):
    architecture =    [# Size: N x C x dim[0] x dim[1]
                       ("conv",         {"in_channels"  : n_channel, 
                                         "out_channels" : n_channel,
                                         "kernel_size"  : 3,
                                         "padding"      : 1,
                                         "stride"       : 1,
                                         "bias"         : False}),
                       ("batchnorm",    {"num_features" : n_channel}),
                       ("ReLU",),
                       # Size: N x C x dim[0] x dim[1]
                       ("conv",         {"in_channels"  : n_channel, 
                                         "out_channels" : n_channel,
                                         "kernel_size"  : 3,
                                         "padding"      : 1,
                                         "stride"       : 1,
                                         "bias"         : False}),
                       ("batchnorm",    {"num_features" : n_channel}),
                      ]
    return (architecture, ("ReLU",))

def conv_down(in_channel, channels, scales, **kw_args):
    architecture =   conv_block(in_channel=in_channel, out_channel=channels[0], scale=scales[0], **kw_args)
    for i in range(1, len(channels)):
        architecture += conv_block(in_channel=channels[i-1], out_channel=channels[i], scale=scales[i], **kw_args)
    
    return architecture

def conv_up(in_channel, channels, scales, **kw_args):
    architecture =   conv_block(in_channel=in_channel, out_channel=channels[0], scale=scales[0], type="transp conv", **kw_args)
    for i in range(1, len(channels)):
        architecture += conv_block(in_channel=channels[i-1], out_channel=channels[i], scale=scales[i], type="transp conv", **kw_args)
    
    return architecture

def build_sequential(architecture):
    if architecture is None:
        return lambda x: x
    
    modules = []
    for layer in architecture:
        if len(layer) == 2:
            name, config = layer
        elif len(layer) == 1:
            name = layer[0]
        else:
            raise RuntimeError("Layer definition ill-formed: {}.".format(layer))
            
        name = name.lower()
        if name == "conv":
            modules.append(torch.nn.Conv2d(**config))
        elif name == "transp conv":
            modules.append(torch.nn.ConvTranspose2d(**config))
        elif name == "linear":
            modules.append(torch.nn.Linear(**config))
        elif name == "leaky relu":
            modules.append(torch.nn.LeakyReLU(config, inplace=True))
        elif name == "relu":
            modules.append(torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            modules.append(torch.nn.PReLU())
        elif name == "tanh":
            modules.append(torch.nn.Tanh())
        elif name == "sigmoid":
            modules.append(torch.nn.Sigmoid())
        elif name == "softplus":
            modules.append(torch.nn.Softplus())
        elif name == "batchnorm":
            modules.append(torch.nn.BatchNorm2d(**config))
        elif name == "residual block":
            modules.append(ResidualBlock(config))
        elif name == "flatten":
            modules.append(Flatten())
        elif name == "unflatten":
            modules.append(UnFlatten(config))
        else:
            raise NotImplementedError("Layer {} not supported yet!".format(name))
    
    return torch.nn.Sequential(*modules)

def merge_aux_label(y, aux_label):
    """Merge aux labels as constant feature maps into y.
    
    Arguments
    ---------
    y : torch.Tensor
        Input tensor. Should be of shape (N,C_y,H,W).
    aux_label : torch.Tensor
        Tensor of labels to be merged. Should have shape (N,C_aux) or (N).
        
    Returns
    -------
    out : torch.Tensor
        Tensor of shape (N,C_y+C_aux,H,W).
    """
    # Assume scalar labels and matching batch size
    if aux_label.dim() == 0 or aux_label.dim() == 1:
        aux_label = aux_label.reshape(-1,1)
    if aux_label.shape[0] != y.shape[0]:
        raise ValueError("aux_label batch size needs to match that of y")
    # Expand aux_label to (N,C,H,W)
    aux = aux_label.reshape(*aux_label.shape, 1, 1)
    aux = aux.expand((*aux_label.shape, *y.shape[-2:]))
    return torch.cat((y, aux), dim=1)