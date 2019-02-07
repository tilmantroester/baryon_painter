import torch

import numpy as np
import collections
import os
import pickle
import argparse

import matplotlib
matplotlib.use('Agg')
    
from baryon_painter.utils import datasets, data_transforms
import baryon_painter.painter
from baryon_painter.models import cvae

pi = np.pi

if __name__ == "__main__":
#     parse = argparse.
    
    data_path = "../../painting_baryons/training_data/BAHAMAS/stacks_new/"
    output_path = "../output/"
    compute_device = "cuda:0"

    n_training_stack = 11
    n_validation_stack = 3
    
    n_scale = 1
    n_aux_label = 1
    label_fields = ["pressure"]
    redshifts = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
#     redshifts = [0.0, 0.125, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    range_compress_transform, range_compress_inv_transform = data_transforms.create_range_compress_transforms(
                                                                    k_values={"dm" : 4.0, 
                                                                              "pressure" : 4}, 
                                                                    modes={"dm":"shift-log",
                                                                           "pressure" : "shift-log"},
#                                                                     k_values={"dm" : (2,1), 
#                                                                               "pressure" : (1,0)}, 
#                                                                     modes={"dm":"x/(1+x)",
#                                                                            "pressure" : "x/(1+x)"},
                                                                    eps=1e-4)

    # range_compress_transform, range_compress_inv_transform = data_transforms.create_range_compress_transforms(
    #                                                                     k_values={"dm" : 1.5, 
    #                                                                               "pressure" : 1}, 
    #                                                                     modes={"dm":"x/(1+x)",
    #                                                                            "pressure" : "1/x"})

    with open(os.path.join(data_path, "train_files_info.pickle"), "rb") as f:
        training_files_info = pickle.load(f)


    transform = data_transforms.chain_transformations([range_compress_transform,
                                                       data_transforms.atleast_3d,
                                                      ])

    inv_transform = data_transforms.chain_transformations([data_transforms.squeeze,
                                                           range_compress_inv_transform,
                                                          ])

    training_dataset = datasets.BAHAMASDataset(files=training_files_info, root_path=data_path,
                                               redshifts=redshifts,
                                               label_fields=label_fields,
                                               n_stack=n_training_stack,
                                               transform=transform,
                                               inverse_transform=inv_transform,
                                               n_feature_per_field=n_scale,
                                               mmap_mode="r",
                                               scale_to_SLICS=True,
                                               subtract_minimum=False
                                              )
    validation_dataset = datasets.BAHAMASDataset(data=training_dataset.data,
                                               redshifts=redshifts,
                                               label_fields=label_fields,
                                               n_stack=n_validation_stack, stack_offset=n_training_stack,
                                               transform=transform,
                                               inverse_transform=inv_transform,
                                               n_feature_per_field=n_scale,
                                               mmap_mode="r",
                                               scale_to_SLICS=True,
                                               subtract_minimum=False
                                              )
    
    n_x_feature = len(training_dataset.label_fields)*n_scale

    dim_x = (n_x_feature, training_dataset.tile_size, training_dataset.tile_size)
    dim_y = (n_scale, training_dataset.tile_size, training_dataset.tile_size)

    dim_z = (1, 16, 16)
    test_net =         {"type" :        "Type-1",
                        "dim_x" :       dim_x,
                        "dim_y" :       dim_y,
                        "dim_z" :       dim_z,
                        "n_x_features": n_x_feature,
                        "aux_label" :   True,
                        "q_x_in" :      cvae.conv_down(in_channel=n_x_feature, channels=[8,16,32], scales=[2,4,4]),
                        "q_y_in" :      cvae.conv_down(in_channel=1+n_aux_label, channels=[8,16,32], scales=[2,4,4]),
                        "q_x_y_out" :     cvae.conv_block(64, 2*dim_z[0], kernel=5)
                                        + [("unflatten", (2, *dim_z)),],
                        "p_y_in" :      None,
                        "p_z_in" :      cvae.conv_up(1, channels=[1,1,1], scales=[2,4,4], bias=False, batchnorm=True),
                        "p_y_z_in" :      cvae.conv_block(n_aux_label+n_scale+1, 16, kernel=5)
                                        + cvae.conv_down(in_channel=16, channels=[32, 64, 128], scales=[2, 2, 2])
                                        + [("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                           ("residual block", cvae.res_block(128)),
                                          ]
                                        + cvae.conv_up(128, channels=[64,32,16], scales=[2,2,2], bias=False, batchnorm=True, activation="ReLU"),
                        "p_y_z_out" :   (# Mu 
                                          cvae.conv_block(16, 8, kernel=7, bias=False, batchnorm=False, activation="ReLU")
                                        + cvae.conv_block(8, n_x_feature, kernel=5, bias=False, batchnorm=False, activation="PReLU")
                                        + cvae.conv_block(n_x_feature, n_x_feature, kernel=3, bias=False, batchnorm=False, activation="softplus"),
    #                                      # Var
#                                           cvae.conv_block(16, 8, kernel=7, bias=False, batchnorm=False, activation="PReLU")
#                                         + cvae.conv_block(8, n_x_feature, kernel=5, bias=False, batchnorm=False, activation="PReLU")
#                                         + cvae.conv_block(n_x_feature, n_x_feature, kernel=3, bias=False, batchnorm=False, activation=None)
                                        ),
                        "min_x_var" :   1e-7,
                        "min_z_var" :   1e-7,
                        "L" :           1,
                        }


    # torch.manual_seed(1234)
    # torch.cuda.manual_seed(1234)

    painter = baryon_painter.painter.CVAEPainter(training_data_set=training_dataset,
                                                 test_data_set=validation_dataset,
                                                 architecture=test_net, 
                                                 compute_device=compute_device)

    def adaptive_batch_size(pepoch, min_batch_size=1, max_batch_size=24):
#         return min(max_batch_size, min_batch_size*2**(pepoch//8))
        steps = [(2, 2), (6, 4), (14, 8), (24, 16), (40, 24)]
        for s in steps[::-1]:
            if pepoch >= s[0]:
                return min(s[1], max_batch_size)
        return min_batch_size

    def adaptive_lr(pepoch):
        step = 24
        min_pepoch = 80-step
        min_lr = 1e-8

        if pepoch < min_pepoch:
            return 1
        else:
            gamma = 0.5
            return max(min_lr, 0.5**((pepoch-min_pepoch)//step))

    run_name = "single_scale_max_z2_no_log_shift_softmax_lr1e-3"
    painter.train(n_epoch=2, n_pepoch=256, learning_rate=1e-3, batch_size=4,
                  adaptive_learning_rate=adaptive_lr, 
                  adaptive_batch_size=adaptive_batch_size,
                  pepoch_size=1568,
                  validation_pepochs=[0,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120], validation_batch_size=8,
                  checkpoint_frequency=10000, statistics_report_frequency=400, 
                  loss_plot_frequency=4000, mavg_window_size=50,
                  show_plots=False,
                  save_plots=True,
                  plot_sample_var=False,
                  plot_power_spectra=["auto", "cross"],
                  plot_histogram=["log"],
                  output_path=os.path.join(output_path, run_name),
                  verbose=True)

