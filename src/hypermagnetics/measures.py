import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def replace_inf_nan(x):
    x = jnp.where(jnp.isinf(x), 0.0, x)
    x = jnp.where(jnp.isnan(x), 0.0, x)
    return x


def huber_loss(target, pred, delta=1.0):
    abs_diff = jnp.abs(target - pred)
    abs_diff = replace_inf_nan(abs_diff)
    return jnp.where(
        abs_diff > delta, delta * (abs_diff - 0.5 * delta), 0.5 * abs_diff**2
    )


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
    sources, r, P, F, F_mt = (
        data["sources"],
        data["r"],
        data["potential"],
        data["field"],
        data["field_mt"],
    )

    # pred = jax.vmap(model, in_axes=(0, None))(sources, r)
    # res = jnp.mean((P - pred) ** 2)

    # potential_loss = jnp.mean(optax.huber_loss(pred, P))

    pred = jax.vmap(model.field, in_axes=(0, None))(sources, r)
    # field_loss = jnp.mean(optax.huber_loss(pred, F))
    field_loss = jnp.mean(optax.huber_loss(pred, F_mt))

    # potential_loss = jnp.mean(huber_loss(pred, P))
    # field_loss = jnp.mean(huber_loss(pred, F))

    res = field_loss

    return res


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
    sources, r, target = data["sources"], data["r"], data["field_mt"]
    # pred = jax.vmap(model, in_axes=(0, None))(sources, r)
    pred = jax.vmap(model.field, in_axes=(0, None))(sources, r)

    diff = target - pred
    diff = replace_inf_nan(diff)
    target = replace_inf_nan(target)

    acc = jnp.linalg.norm(diff, axis=-1) / jnp.linalg.norm(target, axis=-1) * 100
    return jnp.median(acc)
