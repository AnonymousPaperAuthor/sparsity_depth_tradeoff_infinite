## Sparsity-depth Tradeoff in Infinitely Wide Neural Networks

The jupyter notebooks performs kernel regressions with the deep sparse NNGP kernels.

All notebooks require chunGP.py to be imported.
chunGP.py contains functions that load the classification datasets (i.e. MINST, Fashion-MNIST, CIFAR10, CIFAR10-Grayscale), which needs to be locally stored in the same folder.
Also contains functions for

- Generating and save lookup table for sparse NNGP
- Loads the lookup table and obtain the kernel values using the table
- Performs kernel ridge regression

All the paths in the codes needs to be adjusted to a local environment.