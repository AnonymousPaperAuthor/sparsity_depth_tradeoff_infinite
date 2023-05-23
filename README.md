## Sparsity-depth Tradeoff in Infinitely Wide Neural Networks

The jupyter notebooks perform kernel regressions with the deep sparse NNGP kernels.
They generate all plots in the main section and the supplementary section.

All notebooks require chunGP.py to be imported at the beginning.
chunGP.py contains functions that load the classification datasets (i.e. MINST, Fashion-MNIST, CIFAR10, CIFAR10-Grayscale), which needs to be locally stored in the same folder.
Also contains functions for

- Generating and save lookup table for sparse NNGP
- Loads the lookup table and obtain the kernel values using the table
- Performs kernel ridge regression

All the paths in the codes needs to be adjusted to a local environment.

Following packages are required:

- numpy
- jax
- optax
- matplotlib
- scipy
