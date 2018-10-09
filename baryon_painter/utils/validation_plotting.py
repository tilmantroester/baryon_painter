import numpy as np

import matplotlib.pyplot as plt

import cosmotools.power_spectrum_tools
import cosmotools.plotting

pi = np.pi

def plot_samples(output_true, output_pred, input, n_sample=4, 
                 input_label="", 
                 output_labels=[],
                 n_feature_per_field=1,
                 tile_size=1):
    n_row = 2*min(output_true.shape[0], n_sample)
    n_col = (output_true.shape[1]+n_feature_per_field)
    
    fig, ax = plt.subplots(n_row, n_col, sharex=True, sharey=True, figsize=(n_col*tile_size, n_row*tile_size))
    fig.subplots_adjust(hspace=0.02, wspace=0.02*n_col/n_row)
    
    # Different colormaps for different tracers
    imshow_kwargs = [{"cmap" : "viridis", "vmin" : -1, "vmax" : 1},
                     {"cmap" : "magma", "vmin" : -1, "vmax" : 1},
                     {"cmap" : "plasma", "vmin" : -1, "vmax" : 1},
                     {"cmap" : "inferno", "vmin" : -1, "vmax" : 1}]
    # Plot input
    for i in range(min(input.shape[0], n_sample)):
        s = input[i].squeeze()
            
        if n_feature_per_field == 1:
            ax[2*i,0].imshow(s, **imshow_kwargs[0])
            ax[2*i+1,0].axis("off")
        else:
            for j in range(n_feature_per_field):
                ax[2*i,j].imshow(s[j], **imshow_kwargs[0])
                ax[2*i+1,j].axis("off")
        
        
    # Plot output
    for i in range(min(output_true.shape[0], n_sample)):
        for j in range(output_true.shape[1]):
            output_true_plot = output_true[i,j].squeeze()
            output_pred_plot = output_pred[i,j].squeeze()

            if n_feature_per_field == 1:
                ax[2*i,j+1].imshow(output_true_plot, **imshow_kwargs[j+1])
                ax[2*i+1,j+1].imshow(output_pred_plot, **imshow_kwargs[j+1])
            else:
                ax[2*i,j+n_feature_per_field].imshow(output_true_plot, **imshow_kwargs[j//n_feature_per_field+1])
                ax[2*i+1,j+n_feature_per_field].imshow(output_pred_plot, **imshow_kwargs[j//n_feature_per_field+1])
    
    for p in ax.flat:
        p.grid("off")
        p.set_axis_off()

    ax[0,0].set_title(input_label) 
    if output_labels != []:
        for i in range(len(output_labels)):
            ax[0,n_feature_per_field*(i+1)].set_title(output_labels[i])
        
    return fig, ax


def plot_power_spectra(output_true, output_pred, input, L,
                       mode="auto", 
                       output_labels=[], plot_size=(4,2), 
                       input_transform=[],
                       output_transforms=[],
                       n_k_bin=20, logspaced_k_bins=True,
                       n_feature_per_field=1):
    n_row = 2
    n_col = output_true.shape[1]//n_feature_per_field
        
    fig, ax = plt.subplots(n_row, n_col, sharex=True, figsize=(plot_size[0]*n_col, plot_size[1]*n_row))
    if n_col == 1:
        ax = np.atleast_2d(ax).T
        
    fig.subplots_adjust(hspace=0, wspace=0.3)
    
    k_min = 2*pi/L
    k_max = 2*pi/L*output_true.shape[-1]/2
    
    for i in range(n_col):
        for j in range(output_true.shape[0]):
            A_true = output_transforms[j][i](output_true[j,i*n_feature_per_field:(i+1)*n_feature_per_field]).squeeze()
            A_pred = output_transforms[j][i](output_pred[j,i*n_feature_per_field:(i+1)*n_feature_per_field]).squeeze()
            if mode.lower() == "auto":
                B_true = A_true
                B_pred = A_pred
            elif mode.lower() == "cross":
                B_true = input_transform[j](input[j,:n_feature_per_field]).squeeze()
                B_pred = input_transform[j](input[j,:n_feature_per_field]).squeeze()
            else:
                raise ValueError("Invalid mode: {}.".format(mode))
                
            Pk_true, k, Pk_var_true, n_mode = cosmotools.power_spectrum_tools.pseudo_Pofk(A_true, B_true, L, k_min=k_min, k_max=k_max, n_k_bin=n_k_bin, logspaced_k_bins=logspaced_k_bins)
            Pk_pred, k, Pk_var_pred, n_mode = cosmotools.power_spectrum_tools.pseudo_Pofk(A_pred, B_pred, L, k_min=k_min, k_max=k_max, n_k_bin=n_k_bin, logspaced_k_bins=logspaced_k_bins)
            
            ax[0,i].loglog(k, k**2 * Pk_true, alpha=0.5, c="C0", label="")
            ax[0,i].loglog(k, k**2 * Pk_pred, alpha=0.5, c="C1", label="")
            
            ax[1,i].semilogx(k, Pk_pred/Pk_true-1, alpha=0.5, c="C0", label="")
            
    
    for p in ax.flat:
        p.grid("off")
        
    
    if len(output_labels) >= n_col:
        for i in range(n_col):
            ax[0,i].set_title(output_labels[i])
    
    for p in ax[0]:
        p.set_ylabel(r"$k^2 P(k)$")
        p.plot([], [], alpha=0.5, c="C0", label="Truth")
        p.plot([], [], alpha=0.5, c="C1", label="Predicted")
        p.legend()
        
    for p in ax[1]:
        p.set_ylim(-0.5, 0.5)
        p.axhline(0)
        p.set_ylabel("Fractional\ndifference")
        p.set_xlabel(r"k [Mpc$^{-1}$ h]")
        
    if mode.lower() == "auto":
        fig.suptitle("Auto power spectrum")
    else:
        fig.suptitle("Cross power spectrum")
        
    plt.show()
    return fig, ax