import torch
import math

pi = math.pi

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
class UnFlatten(torch.nn.Module):
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

class CVAE(torch.nn.Module):
    def __init__(self, architecture, device="cpu"):
        super().__init__()
                              
        self.device = device
        
        print("CVAE with {} architecture.".format(architecture["type"]))
        self.architecture = architecture
        self.dim_x = architecture["dim_x"]
        self.dim_y = architecture["dim_y"]
        self.dim_z = architecture["dim_z"]
        
        self.L = architecture["L"] if "L" in architecture else 1
                              
        self.n_x_features = architecture["n_x_features"]

        if architecture["type"] == "Type-1":            
            self.q_x_in = build_sequential(architecture["q_x_in"])
            self.q_y_in = build_sequential(architecture["q_y_in"])
            self.q_out = build_sequential(architecture["q_x_y_out"])
                              
            self.p_y_in = build_sequential(architecture["p_y_in"])
            self.p_z_in = build_sequential(architecture["p_z_in"])
            self.p_y_z_in = build_sequential(architecture["p_y_z_in"])
            self.p_mu_out = build_sequential(architecture["p_y_z_out"][0])
            if len(architecture["p_y_z_out"]) > 1:
                self.predict_var = True
                self.p_var_out = build_sequential(architecture["p_y_z_out"][1])
                self.x_var_init_std = architecture["x_var_init_std"] if "x_var_init_std" in architecture else 0.01
                def init_weight(m):
                    if hasattr(m, "weight"):
                        torch.nn.init.normal_(m.weight, std=self.x_var_init_std)
                self.p_var_out.apply(init_weight)
            else:
                self.predict_var = False
                self.p_var_out = None
        else:
            raise NotImplementedError("Architecture {} not supported yet!".format(architecture["type"]))
                
        self.min_z_var = 1e-7
        
        self.alpha_var = 1.0
        self.beta_KL = 1.0        
        
    def Q(self, x, y):
        h_x = self.q_x_in(x)
        h_y = self.q_y_in(y)
        h = torch.cat([h_x, h_y], dim=1)        
        h = self.q_out(h)
        self.z_mu = h[:,0]
        self.z_log_var = h[:,1]

        assert self.z_mu.size()[1:] == self.dim_z, "Dimension of z_mu does not match dim_z: {} vs {}.".format(self.z_mu.size()[1:], self.dim_z)

        eps = torch.randn(size=(self.L, *self.z_mu.size()), device=self.device)
        z = self.z_mu + eps * (torch.exp(self.z_log_var/2) + self.min_z_var)
        return z.view(-1, *self.dim_z)
    
    def P(self, z, y, L=1):
        h_y = self.p_y_in(y)
        h_z = self.p_z_in(z)
        
        h = torch.cat([h_z, h_y.repeat(L, 1, 1, 1)], dim=1)
        h = self.p_y_z_in(h)
        
        x_mu = self.p_mu_out(h)
        assert x_mu.size()[1:] == self.dim_x, "Dimension of x_mu does not match dim_x: {} vs {}.".format(x_mu.size()[1:], self.dim_x)
            
        if self.predict_var:
            x_log_var = self.p_var_out(h)
            assert x_log_var.size()[1:] == self.dim_x, "Dimension of x_log_var does not match dim_x: {} vs {}.".format(x_log_var.size()[1:], self.dim_x)
            return x_mu, x_log_var
        else:
            return x_mu,
        
    def forward(self, x, y):
        z = self.Q(x, y)
        M = x.size(0)
        
        self.KL_term = 0.5/M * torch.sum(self.z_mu**2 + torch.exp(self.z_log_var) - (1 + self.z_log_var))

        params = self.P(z, y, self.L)
        x_mu = params[0]
        self.x_mu = x_mu
        if self.predict_var: 
            log_x_var = params[1]/math.log(self.dim_x[0]*self.dim_x[1]*self.dim_x[2])
            self.x_var = torch.exp(log_x_var)
            self.log_likelihood_fixed_var = -0.5*math.log(2*pi) + (-0.5 * (x.repeat(self.L, 1, 1, 1) - x_mu)**2).sum(3).sum(2).sum(0)/(M*self.L)
            self.log_likelihood_free_var = -0.5*math.log(2*pi) + (-0.5*log_x_var - 0.5*(x.repeat(self.L, 1, 1, 1) - x_mu)**2/self.x_var).sum(3).sum(2).sum(0)/(M*self.L)
            self.log_likelihood =    (1-self.alpha_var)*self.log_likelihood_fixed_var \
                                   + self.alpha_var*self.log_likelihood_free_var   
        else:
            # Fixed variance
            self.log_likelihood = -0.5*math.log(2*pi) + 1/(M*self.L)*(-0.5 * (x.repeat(self.L, 1, 1, 1) - x_mu)**2).sum(3).sum(2).sum(0)

        self.ELBO = -self.KL_term*self.beta_KL + self.log_likelihood.sum()
        return self.ELBO
    
    def sample_P(self, y, return_var=False):
        with torch.no_grad():
            z = torch.randn(size=(y.size(0), *self.dim_z), device=self.device)
            p = self.P(z, y, L=1)
            mu = p[0]
            if len(p) == 2:
                var = torch.exp(p[1])
                if return_var:
                    return mu, var
            
            return mu
                
    def get_stats(self):
        if self.predict_var:
            return (self.ELBO.item(), -self.KL_term.item(), 
                    *self.log_likelihood.detach().cpu().numpy(), 
                    *self.log_likelihood_fixed_var.detach().cpu().numpy(),
                    *self.log_likelihood_free_var.detach().cpu().numpy())
        else:
            return (self.ELBO.item(), -self.KL_term.item(), *self.log_likelihood.detach().cpu().numpy())
    
    def get_stats_labels(self):
        if self.predict_var:
            return    ["ELBO", "KL_term",] \
                    + ["log_likelihood_{}".format(i) for i in range(self.n_x_features)] \
                    + ["log_likelihood_fixed_var_{}".format(i) for i in range(self.n_x_features)] \
                    + ["log_likelihood_free_var_{}".format(i) for i in range(self.n_x_features)]
        else:
            return ["ELBO", "KL_term",] + ["log_likelihood_{}".format(i) for i in range(self.n_x_features)]
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_statistics(self, percentile=0.9):
        params = sorted([(p.numel(), name) for name, p in self.named_parameters() if p.requires_grad], reverse=True)
        n = [p[0] for p in params]
        cumulative = [sum(n[:i+1]) for i in range(len(n))]

        print("Total number of parameters: {}".format(cumulative[-1]))
        print("Top {}\% of all parameters are in the following layers".format(percentile*100))
        for i in range(len(params)):
            if cumulative[i] < cumulative[-1]*0.9:
                print("{:<40s}   {:>8}".format(params[i][1], params[i][0]))
                
    def check_gpu(self):
        for name, p in self.named_parameters():
            if "cuda" not in str(p.data.device):
                print("{} is not on the GPU!".format(name))