import os
import numpy as np
import scipy.ndimage
import scipy.integrate

import pyccl as ccl

import astropy.io.fits as fits

pi = np.pi

def create_y_map(painted_planes, z, resolution, map_size, cosmo, order=3, verbose=True):
    def L_pix(cosmo, chi, theta):
        a = ccl.scale_factor_of_chi(cosmo, chi)
        return chi*a*theta

    def A_pix_mean(cosmo, chi_lo, chi_hi, theta):
        f = lambda chi: L_pix(cosmo, chi, theta)**2
        L = scipy.integrate.quad(f, chi_lo, chi_hi)[0]/(chi_hi-chi_lo)
        return L

    y_map = np.zeros((resolution, resolution))
    
    h = cosmo.params.parameters.h
    d_A = ccl.comoving_angular_distance(cosmo, 1/(1+np.array(z)))
    d_A -= 252.5/h/2
    if d_A[0] < 0:
        d_A[0] = 0 
    d_A = np.append(d_A, d_A[-1] + 252.5/h)

    theta_pix = map_size/resolution*pi/180 # Pixel size in radians
    A_pix_eff = np.array([A_pix_mean(cosmo, d_A[i], d_A[i+1], theta_pix) for i in range(len(z))])
    
    # Low redshift seems to constribute too much
    # Effective distance/redshift might better be estimated by considering volume
    # contributing to that redshift range (i.e., weighted towards higher 
    # redshift, with diminishing effect).
    # d_A_eff = d_A[:-1] # z is already the slice mid-point
    # a_eff = np.array([ccl.scale_factor_of_chi(cosmo, d) for d in d_A_eff])
    
    y_fac = 8.125561e-16 # sigma_T/m_e*c^2 in SI
    mpc = 3.086e22 # m/Mpc
    eV = 1.60218e-19 # Electronvolt in Joules
    cm = 0.01 # Centimetre in metres

    Xe = 1.17
    Xi = 1.08
    
    V_c = (400/h/2048*mpc/cm)**3 # Volume of cell in cm^3
    y_fac = y_fac*eV*mpc**-2 # sigma_T/m_e*c^2 in Mpc^2 eV^-1
    
    # theta_pix = map_size/resolution*pi/180 # Pixel size in radians
    # L_pix = theta_pix*d_A_eff*a_eff # Physical pixel size in Mpc
    
    for i, d in enumerate(painted_planes):
        zoom_factor = resolution/d.shape[0]
        d = d.copy()
        d[np.isnan(d)] = 0
        
        d *= V_c*(Xe+Xi)/Xe*y_fac/A_pix_eff[i]/zoom_factor**2
        if verbose: print(f"z : {z[i]:0.3f}, plane shape: {d.shape}, zoom_factor: {zoom_factor:0.3f}")
        if verbose: print(f"{np.isnan(d).sum()}")
        
        y_map += scipy.ndimage.zoom(d, zoom=zoom_factor, order=order, mode="mirror")
        
    return y_map

def get_tile(m, shift, tile_relative_size, expansion_factor=1):
    n_pixel_m = m.shape[0]
    origin = int(n_pixel_m*shift[0]), int(n_pixel_m*shift[1])
    
    if expansion_factor >= 1:
        n_pixel_tile = int(n_pixel_m*tile_relative_size*expansion_factor)
        offset = int(n_pixel_m*tile_relative_size*(expansion_factor-1)/2)
        tile_corners = ((origin[0]-offset, origin[0]-offset+n_pixel_tile), 
                        (origin[1]-offset, origin[1]-offset+n_pixel_tile))
    else:
        raise ValueError("Expension factors < 1 not supported.")
        
    tile = m.take(range(*tile_corners[0]), axis=0, mode="wrap")\
            .take(range(*tile_corners[1]), axis=1, mode="wrap")
    
    return tile

def make_weight_map(tile_shape, falloff=0.05, sigma=1):
    w = np.ones(tile_shape)
    
    falloff_pixel = int(tile_shape[0]*falloff)
    
    for i in range(falloff_pixel):
        d = falloff_pixel-i
        s = falloff_pixel*sigma
        f = np.exp(-0.5*d**2/s**2)
        w[i] *= f
        w[-i-1] *= f
        w[:,i] *= f
        w[:,-i-1] *= f
        
    return w
    

def generate_tiling(n_pixel_plane, n_pixel_tile, min_tile_overlap=0.5):
    tile_relative_size = n_pixel_tile/n_pixel_plane
    if tile_relative_size < 1-tile_relative_size + tile_relative_size*min_tile_overlap:
        # Overlap less than min_tile_overlap
        A = tile_relative_size*(1-min_tile_overlap)
        B = 1 - 2*tile_relative_size + tile_relative_size*min_tile_overlap
        if B <= A:
            n_tile = 1
        else:
            n_tile = int(np.ceil((B-A)/(tile_relative_size*(1-min_tile_overlap)))) + 1
    else:
        n_tile = 0
    
    tile_origins = np.linspace(0, 1-tile_relative_size, n_tile+2, endpoint=True)    
    tile_slices = []
    for x_shift in tile_origins:
        tile_slices.append([])
        for y_shift in tile_origins:
            x_shift_pixel = int(x_shift*n_pixel_plane)
            y_shift_pixel = int(y_shift*n_pixel_plane)
            
            tile_slice = np.s_[x_shift_pixel:x_shift_pixel+n_pixel_tile,
                               y_shift_pixel:y_shift_pixel+n_pixel_tile]
            tile_slices[-1].append(tile_slice)
    return tile_origins, tile_slices

def process_SLICS(painter, 
                  tile_size, n_pixel_tile, 
                  LOS, z_SLICS, delta_size, delta_path, massplane_path, shifts_path,
                  z_slice,
                  min_tiling_overlap=0.5, verbose=True, 
                  SLICS_density=False,
                  regularise=False,
                  regularise_std=None,
                  return_problematic_tiles=False,
                 ):
    if len(z_SLICS) != len(z_slice):
        raise ValueError("Shapes of z_SLICS and z_slice need to match!")
    
    n_pixel_delta = 7745
    n_pixel_massplane = 4096*3
    massplane_size = 505 # Mpc/h
    
    painted_planes = []
    
    for i in range(len(z_SLICS)):
        if verbose: print(f"Processing z={z_SLICS[i]:.3f}")
        if delta_size[i] < tile_size:
            if verbose: print("  Tile bigger than delta plane, using mass planes.")
            # Get tile from mass plane, then cut out delta map footprint
            shifts = np.loadtxt(os.path.join(shifts_path, f"random_shift_LOS{LOS}"))[::-1]
            projection = lambda idx: ["xy", "xz", "yz"][idx%3]
            massplane_file = os.path.join(massplane_path, f"{z_SLICS[i]:.3f}proj_half_finer_{projection(i)}.dat_LOS{LOS}")
            
            if verbose: print(f"  Loading {massplane_file}.")
            plane = np.fromfile(massplane_file, dtype=np.float32)[1:].reshape(4096*3, -1).T
            # plane -= plane.mean()
            plane *= 1/(3072**3/2/12288**2)
            
            if verbose: print(f"  Extracting tile.")
            tile = get_tile(plane, shift=shifts[i], 
                            tile_relative_size=delta_size[i]/massplane_size, 
                            expansion_factor=tile_size/delta_size[i])
            if SLICS_density:
                tile -= tile.min()
            tile = scipy.ndimage.zoom(tile, zoom=n_pixel_tile/tile.shape[0], mode="mirror")
            
            if verbose: print(f"  Painting on tile.")
            painted_tile = painter.paint(input=tile, 
                                         z=z_slice[i],
                                         transform=True, inverse_transform=True)
            
            painted_plane = get_tile(painted_tile, shift=((1-delta_size[i]/tile_size)/2, (1-delta_size[i]/tile_size)/2),
                                     tile_relative_size=delta_size[i]/tile_size)
            painted_planes.append(painted_plane)
        else:
            if SLICS_density:
                delta_file = os.path.join(delta_path, f"{z_SLICS[i]:.3f}density_LOS{LOS}.fits")
                with fits.open(delta_file) as hdu:
                    delta = hdu[0].data.T
                delta *= 1/(3072**3/2/12288**2)/64
            else:
                # Get tiles from delta map
                delta_file = os.path.join(delta_path, f"{z_SLICS[i]:.3f}delta.dat_bicubic_LOS{LOS}")
                
                delta = np.fromfile(delta_file, dtype=np.float32).reshape(7745, -1).T
                delta += 96 # Mean of massplane
                delta *= 1/(3072**3/2/12288**2)
            
            n_pixel_plane = int(delta_size[i]/tile_size*n_pixel_tile)
            tile_origins, tile_slices = generate_tiling(n_pixel_plane=n_pixel_plane,
                                                        n_pixel_tile=n_pixel_tile,
                                                        min_tile_overlap=0.5)
            
            if verbose: print(f"  Using {len(tile_origins)} tiles (on each side)")
                
            painted_plane = np.zeros((n_pixel_plane, n_pixel_plane))
            weight_plane = np.zeros((n_pixel_plane, n_pixel_plane))
            problematic_tiles = []
            for j, x_shift in enumerate(tile_origins):
                for k, y_shift in enumerate(tile_origins):
                    tile = get_tile(delta, shift=(x_shift, y_shift), 
                                    tile_relative_size=tile_size/delta_size[i])
                    tile = scipy.ndimage.zoom(tile, zoom=n_pixel_tile/tile.shape[0], mode="reflect")
                    if verbose: print(f"    Painting on tile {j+1}-{k+1}")
                    painted_tile = painter.paint(input=tile, 
                                                 z=z_slice[i],
                                                 transform=True, inverse_transform=True)

                    w = make_weight_map(tile.shape, falloff=0.05, sigma=0.5)
                    if regularise_std is not None:
                        if np.any(np.abs(painted_tile-painted_tile.mean()) > painted_tile.std()*regularise_std):
                            problematic_tiles.append((z, tile, painted_tile))
                        if regularise:
                            w[np.abs(painted_tile-painted_tile.mean()) > painted_tile.std()*regularise_std] = 0
                    painted_plane[tile_slices[j][k]] += w*painted_tile
                    weight_plane[tile_slices[j][k]] += w
                    
            painted_planes.append(painted_plane/weight_plane)
            
                    
    if return_problematic_tiles:
        return painted_planes, problematic_tiles
    else:
        return painted_planes
