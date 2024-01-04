import equinox as eqx
import jax
import jax.numpy as jnp
import optax


@eqx.filter_jit
def loss(model, data):
    """
    Calculates the loss function for the given model and data.

    Parameters:
    - model (callable): The model function that takes in sources and grid as inputs and returns predictions.
    - data (dict): A dictionary containing the input data, including sources, grid, and target potential.

    Returns:
    - The mean loss value calculated using the Huber loss function.
    """
    sources, grid, target = data["sources"], data["grid"], data["potential"]
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    return jnp.mean(optax.huber_loss(pred, target))


@eqx.filter_jit
def accuracy(model, data):
    """
    Calculate the median relative error of the model given the data.

    Parameters:
    - model (callable): The model function that takes in sources and grid as inputs and returns predictions.
    - data (dict): A dictionary containing the input data, including sources, grid, and target potential.

    Returns:
    float: The median relative error of the model, as a percentage.

    """
    sources, grid, target = data["sources"], data["grid"], data["potential"]
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    diff = jnp.linalg.norm(target - pred)
    return jnp.median(diff / jnp.linalg.norm(target) * 100)
