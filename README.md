# Scalable magnetic source-to-field inference with hypernetworks

## Overview

This is a set of models for inference from magnetic sources to the scalar potential or the
magnetic field. It uses hypernetworks to act on the source configuration, generating the parameters
for a model that evaluates the potential/field at specified locations.

## Dependencies

- Python 3.x
- jax, equinox, optax
- [for training] wandb
- [for visualisation] matplotlib, ipykernel, pyvista, jupyterlab, trame, ipywidgets
- [for Apple Silicon] ml-dtypes==0.2.0, jax-metal==0.0.4
- [for Linux with NVIDIA GPU] the JAX wheel should be install from URL as shown below

Install via a pyenv or conda environment:

```zsh
conda create -n hypermagnetics python=3.9
conda install -c conda-forge pyvista jupyterlab trame ipywidgets # For visualisation
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # For Linux with NVIDIA GPU
pip install -e .
# For Apple Silicon: pip install -e ".[metal]"
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
