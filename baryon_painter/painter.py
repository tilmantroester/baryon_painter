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
                       architecture="test",
                       compute_device="cpu",
                       ):
        if filename is not None:
            self.load_state_from_file(filename, compute_device)
        else:   
            self.architecture = architecture
            self.compute_device = compute_device
            self.model = models.cvae.CVAE(architecture, torch.device(self.compute_device))
            
        self.training_data = training_data_set
        self.test_data = test_data_set

        
    def load_training_data(self, filename):
        self.data_path = os.path.dirname(filename)
        with open(filename, "rb") as f:
            self.training_data_file_info = pickle.load(f)

    def load_test_data(self, filename):
        self.test_data_path = os.path.dirname(filename)
        with open(filename, "rb") as f:
            self.test_data_file_info = pickle.load(f)

    def train(self, n_epoch=5, n_pepoch=None, learning_rate=1e-4, batch_size=1,
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
        else:
            raise NotImplementedError("Adaptive batch sizes are currently broken.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if adaptive_learning_rate is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=adaptive_learning_rate["step_size"], 
                                                        gamma=adaptive_learning_rate["gamma"])
        else:
            scheduler = None
        
        if self.training_data.n_feature_per_field == 1:
            stats_labels = [l.replace(f"{i}", f) for l in self.model.get_stats_labels() for i, f in enumerate(self.training_data.label_fields)]
        else:
            n_feature_per_field = self.training_data.n_feature_per_field
            stats_labels = self.model.get_stats_labels()
            for j, f in enumerate(self.training_data.label_fields):
                for k in range(n_feature_per_field):
                    for i, l in enumerate(stats_labels):
                        stats_labels[i] = l.replace(f"{j*n_feature_per_field + k}", 
                                                    f"{f}_{k}")
            
             
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            
            model_checkpoint_template = os.path.join(output_path, "checkpoint_epoch{epoch}_batch{batch}_sample{sample}")

            sample_plot_template = os.path.join(output_path, "sample_epoch{}_batch{}_sample{}.png")
            auto_power_plot_template = os.path.join(output_path, "auto_power_epoch{}_batch{}_sample{}.png")
            cross_power_plot_template = os.path.join(output_path, "auto_power_epoch{}_batch{}_sample{}.png")
            histogram_plot_template = os.path.join(output_path, "histogram_epoch{}_batch{}_sample{}.png")
            log_histogram_plot_template = os.path.join(output_path, "log_histogram_epoch{}_batch{}_sample{}.png")
            loss_plot_template = os.path.join(output_path, "loss_epoch{}_batch{}_sample{}.png")
            
            stats_filename = os.path.join(output_path, "stats.txt")
        else:
            model_checkpoint_template = None
            stats_filename = None
            
        stats = TrainingStats(stats_labels, mavg_window_size, 
                              stats_filename=stats_filename)
   
                    
        if show_plots:
            import matplotlib.pyplot as plt
            
        if n_pepoch is None:
            n_pepoch = n_epoch*len(self.training_data)//pepoch_size
            
        n_processed_samples = 0
        n_processed_batches = 0

        last_pepoch_processed_samples = 0
        last_loss_plot = 0
        last_stat_dump = 0
        last_checkpoint_dump = 0
        
        i_pepoch = 0

        for i_epoch in range(n_epoch):
            if verbose: self.model.check_gpu()
            if i_pepoch >= n_pepoch:
                break
                
            for i_batch, batch_data in enumerate(dataloader):

                if n_processed_samples - pepoch_size >= last_pepoch_processed_samples or n_processed_samples == 0:
                    if n_processed_samples != 0:
                        i_pepoch += 1
                        last_pepoch_processed_samples = n_processed_samples
                        if i_pepoch >= n_pepoch:
                            break

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
                                
                with torch.no_grad():
                    stats.push_loss(n_processed_samples, *self.model.get_stats())

                    if n_processed_samples - checkpoint_frequency >= last_checkpoint_dump and model_checkpoint_template is not None:
                        last_checkpoint_dump = n_processed_samples
                        self.save_state_to_file(model_checkpoint_template.format(epoch=i_epoch, batch=i_batch, sample=n_processed_samples))
                        
                    if n_processed_samples - statistics_report_frequency >= last_stat_dump and statistics_report_frequency > 0:
                        last_stat_dump = n_processed_samples
                        
                        print("Epoch: [{}/{}], P-Epoch: [{}/{}], Batch: [{}/{}], Loss: {:.3e}".format(i_epoch, n_epoch, 
                                                                                                      i_pepoch, n_pepoch, 
                                                                                                      i_batch, len(self.training_data)//batch_size,
                                                                                                      stats.loss_terms["ELBO"]["mavg"][-1]))
                        print("Processed batches: {}, processed samples: {}, batch size: {}, learning rate: {}".format(n_processed_batches, n_processed_samples, batch_size,
                                                                                                    " ".join("{:.1e}".format(p["lr"]) for p in optimizer.param_groups)))
                        print(stats.get_pretty_str(n_col=1))
                    
                    if n_processed_samples - loss_plot_frequency >= last_loss_plot and loss_plot_frequency > 0:
                        last_loss_plot = n_processed_samples
                        stats.plot_loss(window_size=200)
                        if show_plots:
                            plt.show()
                        
        self.validate(validation_batch_size=validation_batch_size)
                                                                            
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
                                                 n_feature_per_field=self.test_data.n_feature_per_field,
                                                 tile_size=2.5)
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



    def paint(self, input, input_transform=None, inverse_transform=None):
        if input.shape != (1,*self.model.dim_y):
            raise ValueError(f"Shape mismatch between input and model: {input.shape} vs {self.model.dim_y}")

        with torch.no_grad():
            if input_transform is not None:
                y = input_transform(input)
            else:
                y = input
            y = torch.Tensor(y, device=self.compute_device)
            prediction = self.model.sample_P(y).cpu().numpy()
        
        if inverse_transform is not None:
            return inverse_transform(prediction)
        else:
            return prediction


    def save_state_to_file(self, filename, mode="model_state_dict+metadata"):
        d = {"L"            : self.training_data.L,
             "tile_L"       : self.training_data.tile_L,
             "n_tile"       : self.training_data.n_tile,
             "tile_size"    : self.training_data.tile_size,
             "input_field"  : self.training_data.input_field,
             "label_fields" : self.training_data.label_fields,
            }
        
#         d["transforms"] = [{"z" : z, 
#                            "transforms" : self.training_data.get_transforms(z=z),
#                            "inverse_transforms" : self.training_data.get_inverse_transforms(z=z)}
#                                for z in self.training_data.redshifts]
        
        d["model_architecture"] = self.architecture
        d["model_state_dict"] = self.model.state_dict()
        
        with open(filename, "wb") as f:
            pickle.dump(d, f)
            
            
    def load_state_from_file(self, filename, compute_device="cpu"):
        with open(filename, "rb") as f:
            d = pickle.load(f)
        
        self.compute_device = compute_device
        self.model = models.cvae.CVAE(d["model_architecture"], torch.device(self.compute_device))
        self.model.load_state_dict(d["model_state_dict"])
        
        self.L = d["L"]
        self.tile_L = d["tile_L"]
        self.n_tile = d["n_tile"]
        self.tile_size = d["tile_size"]
        self.input_field = d["input_field"]
        self.label_fields = d["label_fields"]
#         self.transforms = d["transforms"]

class TrainingStats:
    def __init__(self, loss_terms=[], moving_average_window=100, dump_to_file_frequency=10, stats_filename=None):
        self.mavg_window = moving_average_window
        self.n_batches = 0
        self.n_processed_samples = []
        
        self.last_dump_to_file = 0
        self.dump_to_file_frequency = dump_to_file_frequency
        
        self.loss_terms = collections.OrderedDict()
        for loss_term in loss_terms:
            self.loss_terms[loss_term] = {"all" : [], "mavg" : []}       
            
        self.stats_filename = stats_filename
        if self.stats_filename is not None:
            with open(self.stats_filename, "w") as f:
                f.write("# Batch nr, sample nr, {}\n".format(", ".join(loss_terms)))
    
    def push_loss(self, n_sample, *args):
        self.n_batches += 1
        self.n_processed_samples.append(n_sample)
        for i, loss_term in enumerate(self.loss_terms.values()):
            loss_term["all"].append(args[i])
            loss_term["mavg"].append(np.mean(loss_term["all"][-min(self.n_batches, self.mavg_window):]))
        
        if self.n_batches - self.dump_to_file_frequency >= self.last_dump_to_file \
           and self.stats_filename is not None:
            with open(self.stats_filename, "a") as f:
                for s in range(self.last_dump_to_file, self.n_batches):
                    f.write(self.get_str(s) + "\n")
            self.last_dump_to_file = self.n_batches
            
    def get_str(self, idx=-1):
        batch = idx if idx >= 0 else self.n_batches + idx + 1
        s = f"{batch} {self.n_processed_samples[idx]} "
        for loss in self.loss_terms.values():
            s += f"{loss['all'][idx]} "
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