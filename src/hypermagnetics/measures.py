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
def cosine_similarity(model, data):
    sources, r, F = (data["sources"], data["r"], data["field"])
    pred = jax.vmap(model.field, in_axes=(0, None))(sources, r)
    cos_sim = replace_inf_nan(
        1
        - jnp.einsum("...i,...i->...", pred, F[..., :2])
        / (jnp.linalg.norm(pred, axis=-1) * jnp.linalg.norm(F[..., :2], axis=-1))
    )
    return jnp.mean(cos_sim)


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
    sources, r, P, F = (
        data["sources"],
        data["r"],
        data["potential"],
        data["field"],
    )

    pred = jax.vmap(model, in_axes=(0, None))(sources, r)
    potential_loss = jnp.mean(optax.huber_loss(pred, P))

    pred = jax.vmap(model.field, in_axes=(0, None))(sources, r)
    field_loss = jnp.mean(optax.huber_loss(pred, F[..., :2]))

    res = potential_loss + 0.25 * field_loss

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
    sources, r, target = data["sources"], data["r"], data["potential"]
    pred = jax.vmap(model, in_axes=(0, None))(sources, r)
    diff = target - pred

    acc = jnp.linalg.norm(diff) / jnp.linalg.norm(target) * 100
    acc = replace_inf_nan(acc)
    return jnp.median(acc)


@eqx.filter_jit
def accuracy_field(model, data):
    """
    Calculate the median relative error of the model given the data.

    Parameters:
    - model (callable): The model function that takes in sources and grid as inputs and returns predictions.
    - data (dict): A dictionary containing the input data, including sources, grid, and target potential.

    Returns:
    float: The median relative error of the model, as a percentage.

    """
    sources, r, target = data["sources"], data["r"], data["field"]
    pred = jax.vmap(model.field, in_axes=(0, None))(sources, r)
    diff = target[..., :2] - pred

    acc = jnp.linalg.norm(diff, axis=-1) / jnp.linalg.norm(target, axis=-1) * 100
    acc = replace_inf_nan(acc)
    return jnp.median(acc)
