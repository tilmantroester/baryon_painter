import torch
import math

pi = math.pi

from .utils import *

class CVAE(torch.nn.Module):
    def __init__(self, architecture, device="cpu"):
        super().__init__()
                              
        self.device = torch.device(device)
        
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
                self.min_x_var = architecture["min_x_var"] if "min_x_var" in architecture else 1e-7
            else:
                self.predict_var = False
                self.p_var_out = None
            self.use_aux_label = architecture["aux_label"]
            if "prior_z_y" in architecture:
                # Use prior network to get z
                self.prior_network = build_sequential(architecture["prior_z_y"])
            else:
                self.prior_network = None
        else:
            raise NotImplementedError("Architecture {} not supported yet!".format(architecture["type"]))
                
        self.min_z_var = architecture["min_z_var"] if "min_z_var" in architecture else 1e-7
        
        self.likelihood_scaling = architecture["likelihood_scaling"] if "likelihood_scaling" in architecture else 1.0
        self.alpha_var = 1.0
        self.beta_KL = 1.0

        if "cuda" in self.device.type:
            self.cuda()
        
    def sample_z(self, z_mu, z_log_var):
        eps = torch.randn(size=(self.L, *z_mu.size()), device=self.device)
        z = z_mu + eps * (torch.exp(z_log_var/2) + self.min_z_var)
        return z.view(-1, *self.dim_z)
    
    def Q(self, x, y, aux_label=None):
        if aux_label is not None and self.use_aux_label:
            y = merge_aux_label(y, aux_label)
        h_x = self.q_x_in(x)
        h_y = self.q_y_in(y)
        h = torch.cat([h_x, h_y], dim=1)        
        h = self.q_out(h)
        self.z_mu = h[:,0]
        self.z_log_var = h[:,1]

        assert self.z_mu.size()[1:] == self.dim_z, "Dimension of z_mu does not match dim_z: {} vs {}.".format(self.z_mu.size()[1:], self.dim_z)

        return self.sample_z(self.z_mu, self.z_log_var)
    
    def prior(self, y, aux_label=None):
        if self.prior_network is None:
            z_mu =  torch.zeros((y.shape[0], *self.dim_z), device=self.device)
            z_log_var = torch.zeros((y.shape[0], *self.dim_z), device=self.device)
        else:
            if aux_label is not None and self.use_aux_label:
                y = merge_aux_label(y, aux_label)
            h = self.prior_network(y)
            z_mu = h[:,0]
            z_log_var = h[:,1]

        assert z_mu.size()[1:] == self.dim_z, "Dimension of z_mu does not match dim_z: {} vs {}.".format(z_mu.size()[1:], self.dim_z)

        return z_mu, z_log_var
    
    def sample_prior(self, y, aux_label=None):
        with torch.no_grad():
            z_mu, z_log_var = self.prior(y, aux_label)
            return self.sample_z(z_mu, z_log_var)

    
    def P(self, z, y, L=1, aux_label=None):
        if aux_label is not None and self.use_aux_label:
            y = merge_aux_label(y, aux_label)
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
        
    def forward(self, x, y, aux_label=None):
        z = self.Q(x, y, aux_label)
        M = x.size(0)
        
        prior_z_mu, prior_z_log_var = self.prior(y, aux_label)
        prior_z_var = torch.exp(prior_z_log_var)
        
        self.KL_term = 0.5/M * torch.sum((prior_z_mu-self.z_mu)**2/prior_z_var + torch.exp(self.z_log_var)/prior_z_var \
                                          + prior_z_log_var - self.z_log_var - 1)

        params = self.P(z, y, self.L, aux_label)
        x_mu = params[0]
        self.x_mu = x_mu
        if self.predict_var: 
            log_x_var = params[1]#/math.log(self.dim_x[0]*self.dim_x[1]*self.dim_x[2]) * math.log(self.min_x_var)
            self.x_var = torch.exp(log_x_var)
            self.log_likelihood_fixed_var = -0.5*math.log(2*pi) + (-0.5 * (x.repeat(self.L, 1, 1, 1) - x_mu)**2).sum(dim=[3,2,0])/(M*self.L)
            self.log_likelihood_free_var = -0.5*math.log(2*pi) + (-0.5*log_x_var - 0.5*(x.repeat(self.L, 1, 1, 1) - x_mu)**2/self.x_var).sum(dim=[3,2,0])/(M*self.L)
            self.log_likelihood =    (1-self.alpha_var)*self.log_likelihood_fixed_var \
                                   + self.alpha_var*self.log_likelihood_free_var   
        else:
            # Fixed variance
            self.log_likelihood = -0.5*math.log(2*pi) + 1/(M*self.L)*(-0.5 * (x.repeat(self.L, 1, 1, 1) - x_mu)**2).sum(dim=[3,2,0])

        self.ELBO = -self.KL_term*self.beta_KL + self.likelihood_scaling*self.log_likelihood.sum()
        return self.ELBO
    
    def sample_P(self, y, return_var=False, aux_label=None, z=None):
        with torch.no_grad():
            if z is None:
                z = self.sample_prior(y, aux_label)
            else:
                z = torch.tensor(z, device=self.device, dtype=y.dtype)
            p = self.P(z, y, L=1, aux_label=aux_label)
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