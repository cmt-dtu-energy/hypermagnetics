# Scalable magnetic source-to-field inference with hypernetworks

## Overview

This repository contains two implementations: **source-to-field** and **boundary-to-field**.

**Source-to-field** is a set of models for inference from magnetic sources to the scalar potential or the magnetic field. It uses hypernetworks to act on the source configuration, generating the parameters for a model that evaluates the potential/field at specified locations.

**Boundary-to-field** is for estimating the uncertainty in the spatial magnetic field given only observations around a boundary. It is done by contructing a Gaussian process on the boundary, samples from which are then solved exactly using Laplace's equation.

## Dependencies and set-up for source-to-field

- Python 3.9
- jax, equinox, optax
  - [for Apple Silicon] ml-dtypes==0.2.0, jax-metal==0.0.4
  - [for Linux with NVIDIA GPU] the JAX wheel should be install from URL as shown below
- [for training] wandb
- [for visualisation] matplotlib, ipykernel

Install via a pyenv or conda environment:

```zsh
conda create -n hypermagnetics python=3.9 && conda activate hypermagnetics
python -m pip install numpy wheel ml-dtypes==0.2.0
conda install -c conda-forge pyvista jupyterlab trame ipywidgets # For visualisation
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # For Linux with NVIDIA GPU
pip install -e .
# For Apple Silicon: pip install -e ".[metal]"
```

## Dependencies and set-up for boundary-to-field

- Python 3.10
- jax, optax, scipy<1.13
  - Apple Silicon **not supported yet** because jax-metal cannot perform the matrix operations needed for the GPs
  - [for Linux with NVIDIA GPU] the JAX wheel should be install from URL as shown below
- @kstensbo's [custom GP library](https://github.com/kstensbo/dgp/tree/main) for learning a GP given only observations of its gradient
- [for visualisation] matplotlib, ipykernel

```zsh
conda create -n boundary-gp python=3.10 && conda activate boundary-gp
conda install -c conda-forge matplotlib ipykernel # For visualisation
python -m pip install git+https://github.com/berianjames/dgp.git
python -m pip install jax optax scipy==1.12.0 scikit-learn scienceplots
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # For Linux with NVIDIA GPU
```

## Functionality

The jax modelling intentionally eschews classes in favour of functions grouped into modules.

### Source configuration `sources.py`

Sources are represented as a collection of locations in 2D space with associated magnetisations;
the sources are not point-like and are assumed to have a size of 1 unit. A configuration is
generated by:

```python
import hypermagnetics.sources as sources

source_config = {"N": 500, "M": 5, "lim": 3, "res": 32}
configuration = sources.configure(**source_config, key=jr.PRNGKey(40))
```

- `N` (int): Number of samples to generate.
- `M` (int): Number of sources in each sample.
- `lim` (int, optional): Domain range, in units of source radius. Defaults to 3.
- `res` (int, optional): Resolution of the field grid. Defaults to 32.
- `key` (jr.PRNGKey): Random number generator key.

This genererates a 500 samples, each with five sources, in a 32x32 grid over a domain of [-3, 3],
as a dictionary containing `sources`, `grid`, `field` and `potential`.

### Modelling `models.py`

This week's codebase implements an additive hypernetwork for the parameters of an MLP model (`HyperMLP`).

```python
hyperkey, mainkey = jr.split(jr.PRNGKey(42), 2)
model_config = {
    "width": 16,
    "depth": 3,
    "hdepth": 3,
}
model = HyperMLP(**model_config, hyperkey=hyperkey, mainkey=mainkey)
```

- width `int`: Width of the inference MLP.
- depth `int`: Depth of the inference MLP.
- hdepth `int`: Depth of the hypernetwork; its width is fixed to the number of parameters needed.
- hyperkey `jr.PRNGKey`: Random number generator key for the hypernetwork.
- mainkey `jr.PRNGKey`: Random number generator key for the inference network.

The width and depth are for the *inference* network. The hypernetwork that generates the inference network parameters has width equal to the number of parameters needed, and depth equal to `hdepth`.

To call the model, we need to specify the source configuration and the locations at which to evaluate the field/potential:

```python
sources, locations = configuration["sources"], configuration["grid"]
pred = jax.vmap(model, in_axes=(0, None))(sources, locations)
```

The `vmap` distributes the (arbitrary number of) sources to the hypernetwork, summing the network's output across the source dimension. These aggregate parameters are the parameters for the inference network, which then evaluates the potential/field at the specified locations.

### Training `train.py`

The training script automates runs with the above modules. It generates a source configuration, creates a model, and trains it to minimise the mean squared error between the predicted and true field.

The Huber loss is used to reduce the influence of outliers in the field. For evaluating the accuracy, the median of the relative error is used. The training script logs the training and validation losses to wandb.
