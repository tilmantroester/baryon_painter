# Baryon Painter

The baryon painter is a set of deep generative models (currently a CVAE and CGAN) that maps matter density to pressure. The results are described in [T. Tröster et al, 2019](https://www.arxiv.org/abs/1903.12173).

![Sample tiles](https://github.com/tilmantroester/baryon_painter/raw/master/notebooks/plots/samples_z0.0_z1.0.png)

# Usage

Both the CVAE and CGAN expose a simple interface, the `paint` method, that takes a dark matter density tile and its redshift as input and generates the corresponding pressure tile:

```python
import baryon_painter.painter

model_path = "trained_models/CVAE/fiducial/"

painter = baryon_painter.painter.CVAEPainter((os.path.join(model_path, "model_state"),
                                              os.path.join(model_path, "model_meta")))
                                             
pressure_tile_generated = painter.paint(input=dm_tile, 
                                        z=redshift_of_tile, 
                                        transform=True, inverse_transform=True)
```

An example notebook (used to make the plot of the sample tiles above) can be found here: [here](https://github.com/tilmantroester/baryon_painter/blob/master/notebooks/validation_plots.ipynb)

Other potential useful scipts"
* the training script for the CVAE: [here](https://github.com/tilmantroester/baryon_painter/blob/master/scripts/CVAE_single_scale.py)
* the script for the light-cone generation: [here](https://github.com/tilmantroester/baryon_painter/blob/master/scripts/create_lightcone.py)

The architectures and training schedules of the fiducial models are described [here](https://github.com/tilmantroester/baryon_painter/blob/master/trained_models/README.md).
