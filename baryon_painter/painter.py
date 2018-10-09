import os
import pickle
import collections

import numpy as np

import torch
import torch.utils.data

from baryon_painter.utils import validation_plotting
from baryon_painter import models

class Painter:
    """Abstract base class for a baryon painter.

    This class should be sub-classed and the methods ``load_state`` and 
    ``paint`` implemented.
    """

    def __init__(self):
        raise NotImplementedError("This is an abstract base class.")

    def load_state_from_file(self, filename):
        raise NotImplementedError("This is an abstract base class.")

    def paint(self, input):
        raise NotImplementedError("This is an abstract base class.")


class CVAEPainter(Painter):
    def __init__(self, filename=None,
                       training_data_set=None, test_data_set=None,
                       tile_size=512,
                       architecture="test",
                       compute_device="cpu",
                       ):
        self.training_data = training_data_set
        self.test_data = test_data_set

        self.compute_device = compute_device

        self.model = models.cvae.CVAE(architecture, torch.device(self.compute_device))
        
    def load_training_data(self, filename):
        self.data_path = os.path.dirname(filename)
        with open(filename, "rb") as f:
            self.training_data_file_info = pickle.load(f)

    def load_test_data(self, filename):
        self.test_data_path = os.path.dirname(filename)
        with open(filename, "rb") as f:
            self.test_data_file_info = pickle.load(f)

    def train(self, n_epoch=64, learning_rate=1e-4, batch_size=1,
                    adaptive_learning_rate=None, adaptive_batch_size=None,
                    validation_pepochs=[0, 1], validation_batch_size=4,
                    checkpoint_frequency=1000, statistics_report_frequency=50, 
                    loss_plot_frequency=1000, mavg_window_size=20,
                    show_plots=True,
                    output_path=None,
                    verbose=True,
                    pepoch_size=3136,
                    var_anneal_fn=None, KL_anneal_fn=None):
        """Train. We use pseudo epoch as a unit of training time with 
        1 pepoch = 3136 samples and 64 pepoch = 1 epoch (assuming 4x4 tiling of the stacks)."""
        
        if self.training_data is None:
            raise RuntimeError("Trying to train but no training data specified.")
        
        if len(validation_pepochs) > 0 and self.test_data is None:
            raise RuntimeError("Trying to validate but no test data specified.")            


        if adaptive_batch_size is None and batch_size > 0:
            dataloader = torch.utils.data.DataLoader(self.training_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if adaptive_learning_rate is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=adaptive_learning_rate["step_size"], 
                                                        gamma=adaptive_learning_rate["gamma"])
        else:
            scheduler = None
            
        stats_labels = [l.replace(f"{i}", f) for l in self.model.get_stats_labels() for i, f in enumerate(self.training_data.label_fields)]
        stats = TrainingStats(stats_labels, mavg_window_size)
        
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            
            model_checkpoint_template = os.path.join(output_path, "checkpoint_epoch{}_batch{}_sample{}")

            sample_plot_template = os.path.join(output_path, "sample_epoch{}_batch{}_sample{}.png")
            auto_power_plot_template = os.path.join(output_path, "auto_power_epoch{}_batch{}_sample{}.png")
            cross_power_plot_template = os.path.join(output_path, "auto_power_epoch{}_batch{}_sample{}.png")
            histogram_plot_template = os.path.join(output_path, "histogram_epoch{}_batch{}_sample{}.png")
            log_histogram_plot_template = os.path.join(output_path, "log_histogram_epoch{}_batch{}_sample{}.png")
            loss_plot_template = os.path.join(output_path, "loss_epoch{}_batch{}_sample{}.png")
            
            stats_file = open(os.path.join(output_path, "stats.txt"), "w")
            stats_file.write("# Batch nr (batch size={}), sample nr, {}\n".format(batch_size, " ,".join(stats_labels)))
        
        n_processed_samples = 0
        n_processed_batches = 0

        last_pepoch_processed_samples = 0
        last_loss_plot = 0
        last_stat_dump = 0
        i_pepoch = 0

        for i_epoch in range(n_epoch):
            if verbose: self.model.check_gpu()
            
            for i_batch, batch_data in enumerate(dataloader):

                if n_processed_samples - pepoch_size >= last_pepoch_processed_samples or n_processed_samples == 0:
                    i_pepoch += 1
                    last_pepoch_processed_samples = n_processed_samples

                    if callable(var_anneal_fn):
                        self.model.alpha_var = var_anneal_fn(i_pepoch)
                    if callable(KL_anneal_fn):
                        self.model.beta_KL = KL_anneal_fn(i_pepoch)
                    if scheduler is not None:
                        scheduler.step()
                        
                    if adaptive_batch_size is not None:
                        batch_size = adaptive_batch_size(i_pepoch)
                        dataloader = torch.utils.data.DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
                        
                    if i_pepoch in validation_pepochs:
                        self.validate(validation_batch_size=validation_batch_size)

                x = torch.cat(batch_data[0][1:], dim=1).to(self.model.device)
                y = batch_data[0][0].to(self.model.device)
                
                ELBO = self.model(x, y)
                
                optimizer.zero_grad()
                (-ELBO).backward()
                optimizer.step()
                
                n_processed_samples += x.size(0)
                n_processed_batches += 1
                
                stats.push_loss(n_processed_samples, *self.model.get_stats())
                
                with torch.no_grad():
                    if output_path is not None:
                        stats_file.write(stats.get_str() + "\n")
                        stats_file.flush()
                            
                    if n_processed_samples - statistics_report_frequency >= last_stat_dump and statistics_report_frequency > 0:
                        last_stat_dump = n_processed_samples
                        
                        print("Epoch: [{}/{}], P-Epoch: [{}/{}], Batch: [{}/{}], Loss: {:.3e}".format(i_epoch, n_epoch, 
                                                                                                      i_pepoch, n_epoch*len(self.training_data)//pepoch_size, 
                                                                                                      i_batch, len(self.training_data)//batch_size,
                                                                                                      stats.loss_terms["ELBO"]["mavg"][-1]))
                        print("Processed batches: {}, processed samples: {}, batch size: {}, learning rate: {}".format(n_processed_batches, n_processed_samples, batch_size,
                                                                                                    " ".join("{:.1e}".format(p["lr"]) for p in optimizer.param_groups)))
                        print(stats.get_pretty_str(n_col=3))
                    
                    if n_processed_samples - loss_plot_frequency >= last_loss_plot and loss_plot_frequency > 0:
                        last_loss_plot = n_processed_samples
                        stats.plot_loss(window_size=200)
                        if show_plots:
                            plt.show()
                        
        self.validate(validation_batch_size=validation_batch_size)
                                                    
        stats_file.close()
                        
        return stats

    def validate(self, validation_batch_size=8,
                       plot_samples=1, plot_power_spectra="auto", plot_histogram="log", show_plots=True):
        validation_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=validation_batch_size, shuffle=True)

        with torch.no_grad():
            batch_data = next(validation_dataloader.__iter__())
            x = torch.cat(batch_data[0][1:], dim=1).to(self.model.device)
            y = batch_data[0][0].to(self.model.device)
            x_pred = self.model.sample_P(y)

            indicies = batch_data[1].numpy()
            inverse_transforms = [self.test_data.get_inverse_transforms(idx) for idx in indicies]
            if plot_samples > 0:
                fig, _ = validation_plotting.plot_samples(output_true=x.cpu().numpy(), 
                                                 input=y.cpu().numpy(), 
                                                 output_pred=x_pred.cpu().numpy(),
                                                 n_sample=plot_samples,
                                                 input_label=self.test_data.input_field,
                                                 output_labels=self.test_data.label_fields,
                                                 n_feature_per_field=self.test_data.n_feature_per_field)
                if show_plots:
                    fig.show()

            if plot_power_spectra is not None:
                fig, _ = validation_plotting.plot_power_spectra(output_true=x.cpu().numpy(), 
                                                       input=y.cpu().numpy(), 
                                                       output_pred=x_pred.cpu().numpy(),
                                                       L=self.test_data.tile_L,
                                                       output_labels=self.test_data.label_fields,
                                                       mode=plot_power_spectra,
                                                       input_transform=[t[0] for t in inverse_transforms],
                                                       output_transforms=[t[1:] for t in inverse_transforms],
                                                       n_feature_per_field=self.test_data.n_feature_per_field)
                if show_plots:
                    fig.show()



    def paint(self, input, fields=["pressure"]):
        if input.shape != self.model.dim_y:
            raise ValueError(f"Shape mismatch between input and model: {input.shape} vs {self.model.dim_y}")

        with torch.no_grad():
            y = self.data_transform["dm"](input)
            y = torch.Tensor(y, device=self.compute_device)
            prediction = self.model.sample_P(y).numpy()
        return self.inverse_data_transform(prediction)


    def load_state_from_file(self, filename):
        self.model = torch.load(filename)

class TrainingStats:
    def __init__(self, loss_terms=[], moving_average_window=100):
        self.mavg_window = moving_average_window
        self.n_batches = 0
        self.n_processed_samples = []
        
        self.loss_terms = collections.OrderedDict()
        for loss_term in loss_terms:
            self.loss_terms[loss_term] = {"all" : [], "mavg" : []}        
    
    def push_loss(self, n_sample, *args):
        self.n_batches += 1
        self.n_processed_samples.append(n_sample)
        for i, loss_term in enumerate(self.loss_terms.values()):
            loss_term["all"].append(args[i])
            loss_term["mavg"].append(np.mean(loss_term["all"][-min(self.n_batches, self.mavg_window):]))
            
    def get_str(self):
        s = "{} {} ".format(self.n_batches, self.n_processed_samples[-1])
        for loss in self.loss_terms.values():
            s += "{} ".format(loss["all"][-1])
        return s
        
    def get_pretty_str(self, n_col=1):
        s = ""
        max_len_key = max([len(key) for key in self.loss_terms.keys()])
        items_per_row = 0
        for i, (key, term) in enumerate(self.loss_terms.items()):
            s += "{key:<{width}s}: {value:8.3e}     ".format(key=key, width=max_len_key, value=term["mavg"][-1])
            items_per_row += 1
            if items_per_row >= n_col:
                s += "\n"
                items_per_row = 0
        return s

    def plot_loss(self, loss_term="ELBO", window_size=100):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        
        n = self.n_batches
        if n > 500:
            total_loss = self.loss_terms[loss_term]["all"][::n//500]
            total_loss_mavg = self.loss_terms[loss_term]["mavg"][::n//500]
        else:
            total_loss = self.loss_terms[loss_term]["all"]
            total_loss_mavg = self.loss_terms[loss_term]["mavg"]
        
        
        ax[0].semilogy(np.linspace(1, n, len(total_loss)), np.abs(total_loss), alpha=0.5, label="{}".format(loss_term))
        ax[0].semilogy(np.linspace(self.mavg_window, n, len(total_loss_mavg)), np.abs(total_loss_mavg), label="{} mavg".format(loss_term))
        ax[0].legend()
        
        x_range = np.linspace(max(n-window_size,1), n, min(n, window_size))
        ax[1].plot(x_range, total_loss[-min(n, window_size):], alpha=0.5, label="{}".format(loss_term))
        ax[1].plot(x_range, total_loss_mavg[-min(n, window_size):], label="{} mavg".format(loss_term))
        ax[1].legend()
        ax[1].set_ylim(min(total_loss[-min(n, window_size):]), max(total_loss[-min(n, window_size):]))
    
        return fig, ax