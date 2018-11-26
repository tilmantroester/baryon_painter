import numpy as np

from baryon_painter.process_SLICS import get_tile, generate_tiling, make_weight_map
pi = np.pi

def check_get_tile():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    import pyccl as ccl
    import cosmotools.plotting as plotting
    
    # WMAP9
    Omega_m = 0.2905
    Omega_b = 0.0473
    Omega_L = 0.7095
    h = 0.6898
    sigma_8 = 0.826
    n_s = 0.969
    cosmo = ccl.Cosmology(Omega_c=(1-Omega_L-Omega_b), Omega_b=Omega_b, Omega_k=0,
                          h=h, sigma8=sigma_8, n_s=n_s)

    z_SLICS = 0.042
    d_A_SLICS = ccl.comoving_angular_distance(cosmo, 1/(1+z_SLICS))*h # units of Mpc/h

    
    shifts = np.loadtxt("../data/training_data/SLICS/random_shift_LOS1097")[::-1]

    tile_relative_size = d_A_SLICS*10/180*pi/505

    plane = np.fromfile(f"../data/training_data/SLICS/massplanes/{z_SLICS:.3f}proj_half_finer_xy.dat_LOS1097", dtype=np.float32)[1:].reshape(4096*3, -1).T
    plane *= 64
    plane -= plane.min()

    delta = np.fromfile(f"../data/training_data/SLICS/delta/{z_SLICS:.3f}delta.dat_bicubic_LOS1097", dtype=np.float32).reshape(7745, -1).T
    delta *= 64
    delta -= delta.min()
    delta = delta[::7, ::7]

    expansion_factor = 1.5
    tile = get_tile(plane, shifts[0], tile_relative_size, expansion_factor=expansion_factor)

    delta_region = patches.Rectangle(xy=(0,0), width=1, height=1, fill=False)

    # Plot stuff
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im = ax[0].imshow(np.log(tile+1), extent=(-(expansion_factor-1)/2, 1.0+(expansion_factor-1)/2, 
                                              -(expansion_factor-1)/2, 1.0+(expansion_factor-1)/2))
    plotting.subplot_colorbar(im, ax[0])
    ax[0].add_artist(delta_region)

    im = ax[1].imshow(np.log(delta+1), extent=(0.0, 1.0, 0.0, 1.0))
    plotting.subplot_colorbar(im, ax[1])
    
    ax[0].set_title("Mass plane")
    ax[1].set_title("Delta")
    fig.suptitle("Extract tiles from SLICS")

def check_make_weight_map():
    import matplotlib.pyplot as plt
    import cosmotools.plotting as plotting
    
    w = make_weight_map((512, 512), falloff=0.05, sigma=0.5)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(w)
    plotting.subplot_colorbar(im, ax)
    
    ax.set_title("Weight map, 512x512, falloff=0.05, sigma=0.5")

def test_generate_tiling(plot=False):
    origins, tiles = generate_tiling(512, 256, min_tile_overlap=0.0)
    assert len(origins) == 2
    origins, tiles = generate_tiling(512, 250, min_tile_overlap=0.0)
    assert len(origins) == 3
    origins, tiles = generate_tiling(512, 256, min_tile_overlap=0.5)
    assert len(origins) == 3

    origins, tiles = generate_tiling(512, 128, min_tile_overlap=0.0)
    assert len(origins) == 4
    
    _, tiles = generate_tiling(512, 32, min_tile_overlap=0.33)
    w = np.zeros((512, 512))
    for t in tiles:
        for s in t:
            w[s] += 1

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.imshow(w)

if __name__ == "__main__":
    check_get_tile()