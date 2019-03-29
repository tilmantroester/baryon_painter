# Trained models

## CVAE

We train the CVAE by first increasing the minibatch size from 4 to 24 while keeping the learning rate fixed at 0.001 and then decrease the learning rate to 1e-6 in powers of 0.1. 
The ELBO on the training set starts to diverge from that on the test set after about 100000 samples, corresponding to about a third of all samples in the training set. 
The samples are not independent however, since they are formed by combinations of two tiles, as described in Sec. 2.2 in the paper. 
The number of independent samples that can therefore be formed from the 11 slices, 16 tiles, and 11 redshifts is 1936. 
The expected number of samples required to visit those independent samples is approximately n log n = 14652, such that seeing signs of overfitting before the whole training set has been processed is not unexpected.
We stop the training after 150000 samples, which corresponds to a training time of about 3 hours on a single Nvidia GTX 1080 Ti.

### Recognition network

The recognition network `q(z|x,y)` takes as input
* x, the pressure tile (shape (N,1,512,512))
* y, the dark matter tile and its redshift (shape (N,2,512,512))

Part A of the network processes x, Part B processes y. The output of the two get concatenated and fed through Part C. The output are the mean and log variance of the latent variable z (shape (N,1,16,16) for both).

#### Part A

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |1/8               |4       |2       |F     |T          | ReLU       |
|Conv      |8/16              |8       |4       |F     |T          | ReLU       |
|Conv      |16/32             |8       |4       |F     |T          | ReLU       |

#### Part B

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |2/8               |4       |2       |F     |T          | ReLU       |
|Conv      |8/16              |8       |4       |F     |T          | ReLU       |
|Conv      |16/32             |8       |4       |F     |T          | ReLU       |

#### Part C

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |64/2              |5       |1       |F     |T          | ReLU       |

### Prior network

The prior network `p(z|y)` takes as input
* y, the dark matter tile and its redshift (shape (N,2,512,512))

The output are the mean and log variance of the latent variable z (shape (N,1,16,16) for both). The architecture is basically Part B and Part C from the recognition network.

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |2/8               |4       |2       |F     |T          | ReLU       |
|Conv      |8/16              |8       |4       |F     |T          | ReLU       |
|Conv      |16/32             |8       |4       |F     |T          | ReLU       |
|Conv      |32/2              |5       |1       |F     |T          | ReLU       |

### Generator network

The generator network `p(x|y,z)` takes as input
* y, the dark matter tile and its redshift (shape (N,2,512,512))
* z, the latent variable (shape (N,1,16,16))

The latent variable is passed to Part A of the network. The output of Part A is concatenated with y and passed through Part B. The output of Part B is the mean of the predidicted pressure tile with shape (N,1,512,512).

#### Part A

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |1/1               |4       |2       |F     |T          | ReLU       |
|Conv      |1/1               |8       |4       |F     |T          | ReLU       |
|Conv      |1/1               |8       |4       |F     |T          | ReLU       |

#### Part B

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |3/16              |5       |1       |F     |T          | ReLU       |
|Conv      |16/32             |4       |2       |F     |T          | ReLU       |
|Conv      |32/64             |4       |2       |F     |T          | ReLU       |
|Conv      |64/128            |4       |2       |F     |T          | ReLU       |
|ResBlock x 4 |               |        |        |      |           | ReLU       |
|ConvTransp|128/64            |4       |2       |F     |T          | ReLU       |
|ConvTransp|64/32             |4       |2       |F     |T          | ReLU       |
|ConvTransp|32/16             |4       |2       |F     |T          | ReLU       |
|ConvTransp|16/8              |7       |1       |F     |F          | PReLU      |
|ConvTransp|8/1               |5       |1       |F     |F          | PReLU      |
|ConvTransp|1/1               |3       |1       |F     |F          | Softplus   |

#### Residual block

| Layer    | Channel (in/out) | Kernel | Stride | Bias | BatchNorm | Activation |
|----------|------------------|--------|--------|------|-----------|------------|
|Conv      |128/128           |3       |1       |F     |T          | ReLU       |
|Conv      |128/128           |3       |1       |F     |T          |            |

