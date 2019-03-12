import os
import glob
import argparse

import numpy as np
import pyccl as ccl

import baryon_painter.process_SLICS

pi = np.pi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="CVAE")
    parser.add_argument("--CVAE-path")

    parser.add_argument("--CGAN-module-path")
    parser.add_argument("--CGAN-parts-path")
    parser.add_argument("--CGAN-checkpoint")

    parser.add_argument("--SLICS-base-path", required=True)
    parser.add_argument("--SLICS-LOS", required=True)

    parser.add_argument("--n-plane", default=15)
    parser.add_argument("--tile-overlap", default=0.2)

    parser.add_argument("--output-resolution", default=7745//5)

    parser.add_argument("--output-file", required=True)
    parser.add_argument("--output-file-planes")

    args = parser.parse_args()

    if args.model_type == "CVAE":
        print("Using CVAE.")
        import baryon_painter.painter
        cvae_base_path = args.CVAE_path
        painter = baryon_painter.painter.CVAEPainter((os.path.join(cvae_base_path, "model_state"),
                                                      os.path.join(cvae_base_path, "model_meta")))
    elif args.model_type == "CGAN":
        print("Using GAN")
        gan_module_path = args.CGAN_module_path

        import sys
        sys.path.append(gan_module_path)
        from src.tools.template import GAN_Painter

        parts_folder = args.CGAN_parts_path
        checkpoint = args.CGAN_checkpoint

        painter = GAN_Painter(parts_folder, 
                              checkpoint_file=checkpoint,
                              device="cpu")
    else:
        parser.error("Only CVAE and CGAN are supported for --model-type.")


    SLICS_base_path = args.SLICS_base_path
    LOS = int(args.SLICS_LOS)
    output_file = args.output_file

    print(f"Looking in {SLICS_base_path} for SLICS files.")
    print(f"Processing LOS{LOS}.")
    print(f"Writing result to {output_file}.")
    delta_path = os.path.join(SLICS_base_path, "delta")
    massplane_path = os.path.join(SLICS_base_path, "massplanes")
    shifts_path= os.path.join(SLICS_base_path, "random_shifts")

    delta_filenames = glob.glob(os.path.join(delta_path, f"*delta.dat_bicubic_LOS{LOS}"))

    # These are the redshifts of the mid-points of the slices
    z_SLICS = [float(z[:z.find("delta")]) for z in [os.path.split(f)[1] for f in delta_filenames]]
    z_SLICS = np.array(sorted(z_SLICS))

    print("SLICS redshifts:", z_SLICS)

    Omega_m = 0.2905
    Omega_b = 0.0473
    Omega_L = 0.7095
    h = 0.6898
    sigma_8 = 0.826
    n_s = 0.969
    cosmo_SLICS = ccl.Cosmology(Omega_c=(1-Omega_L-Omega_b), Omega_b=Omega_b, Omega_k=0,
                                h=h, sigma8=sigma_8, n_s=n_s, m_nu=0.0)

    d_A_SLICS = ccl.comoving_angular_distance(cosmo_SLICS, 1/(1+z_SLICS))*h # units of Mpc/h

    # Physical redshift of the slices
    z_slice = np.array([1/ccl.scale_factor_of_chi(cosmo_SLICS, 252.5/h*i) - 1 for i in range(len(z_SLICS))])

    n_z = int(args.n_plane)
    tile_overlap = float(args.tile_overlap)

    print(f"Painting {n_z} out of {len(z_SLICS)} planes.")
    print(f"Using an overlap of {tile_overlap}.")

    painted_planes = baryon_painter.process_SLICS.process_SLICS(
                                   painter, 
                                   tile_size=100.0, n_pixel_tile=512,
                                   LOS=LOS, 
                                   z_SLICS=z_SLICS[:n_z], delta_size=d_A_SLICS[:n_z]*10/180*pi, 
                                   delta_path=delta_path, 
                                   massplane_path=massplane_path, 
                                   shifts_path=shifts_path,
                                   z_slice=z_slice[:n_z],
                                   min_tiling_overlap=tile_overlap,
                                   regularise=False,
                                   regularise_std=None
                                )

    output_resolution = int(args.output_resolution)
    y_map = baryon_painter.process_SLICS.create_y_map(painted_planes, z_SLICS[:n_z], 
                              resolution=output_resolution, map_size=10.0, cosmo=cosmo_SLICS, order=5)

    np.save(output_file, y_map)
    if args.output_file_planes is not None:
        import pickle
        with open(args.output_file_planes, "wb") as f:
            pickle.dump(painted_planes, f)

    
    
