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
    sources, r, P, F = data["sources"], data["r"], data["potential"], data["field"]

    pred = jax.vmap(model, in_axes=(0, None))(sources, r)
    potential_loss = jnp.mean(optax.huber_loss(pred, P))

    pred = jax.vmap(model.field, in_axes=(0, None))(sources, r)
    field_loss = jnp.mean(optax.huber_loss(pred, F))

    return potential_loss + field_loss


@eqx.filter_jit
def cached_loss(model, data):
    """
    Calculates the loss function for the given model and data, using a caching for the parameters. This is
    used for example with the Fourier model, where the expensive wavenumber computation can be cached.

    Parameters:
    - model (callable): The model function that takes in sources and grid as inputs and returns predictions.
    - data (dict): A dictionary containing the input data, including sources, grid, and target potential.

    Returns:
    - The mean loss value calculated using the Huber loss function.
    """
    sources, _, target = data["sources"], data["r"], data["potential"]
    pred = jax.vmap(model.cached_evaluation)(*jax.vmap(model.prepare_weights)(sources))
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
    sources, r, target = data["sources"], data["r"], data["potential"]
    pred = jax.vmap(model, in_axes=(0, None))(sources, r)
    diff = jnp.linalg.norm(target - pred)
    return jnp.median(diff / jnp.linalg.norm(target) * 100)
