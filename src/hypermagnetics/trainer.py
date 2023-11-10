import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import hypermagnetics.sources as sources
from hypermagnetics.models import HyperMLP

optim = optax.adam(learning_rate=0.001)


def loss(model, sources, grid, target):
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    return jnp.mean(optax.huber_loss(pred, target))


@eqx.filter_jit
def step(model, opt_state, sources, grid, target):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, sources, grid, target)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def accuracy(model, data):
    sources, grid, target = data["sources"], data["grid"], data["potential"]
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    diff = jnp.linalg.norm(target - pred)
    return jnp.median(diff / jnp.linalg.norm(target) * 100)


if __name__ == "__main__":
    train_config = {"N": 500, "M": 1, "key": jr.PRNGKey(40), "lim": 3, "res": 100}
    test_config = {"N": 500, "M": 1, "key": jr.PRNGKey(41), "lim": 3, "res": 100}
    train = sources.configure(**train_config)
    test = sources.configure(**test_config)

    epochs = 100
    logger = {"train_loss": [], "train_acc": [], "val_acc": []}
    model = HyperMLP(16, 3, *jr.split(jr.PRNGKey(42), 2))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for epoch in range(epochs):
        model, opt_state, train_loss = step(
            model,
            opt_state,
            train["sources"],
            train["grid"],
            train["potential"],
        )

        if (epoch % (epochs / 100)) == 0:
            logger["train_loss"].append(train_loss)
            logger["train_acc"].append(accuracy(model, train))
            logger["val_acc"].append(accuracy(model, test))

        if (epoch % (epochs / 100)) == 0:
            print(
                f"{epoch=}, train_loss={train_loss:.4f}, ",
                f"accuracy={logger['train_acc'][-1]:.4f}",
                f"val_accuracy={logger['val_acc'][-1]:.4f}",
            )
