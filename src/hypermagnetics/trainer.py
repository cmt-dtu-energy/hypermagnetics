import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import hypermagnetics.sources as sources
import wandb
from hypermagnetics.models import HyperMLP


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
    source_config = {"N": 500, "M": 3, "lim": 3, "res": 32}
    train = sources.configure(**source_config, key=jr.PRNGKey(40))
    val = sources.configure(**source_config, key=jr.PRNGKey(41))

    hyperkey, mainkey = jr.split(jr.PRNGKey(42), 2)
    model_config = {
        "width": 32,
        "depth": 3,
        "hdepth": 2,
    }
    model = HyperMLP(**model_config, hyperkey=hyperkey, mainkey=mainkey)

    trainer_config = {"learning_rate": 0.001, "epochs": 1000}
    optim = optax.adam(trainer_config["learning_rate"])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    run = wandb.init(
        project="hypermagnetics",
        config={
            "trainer_config": trainer_config,
            "model_config": model_config,
            "source_config": source_config,
        },
    )

    for epoch in range(trainer_config["epochs"]):
        model, opt_state, train_loss = step(
            model,
            opt_state,
            train["sources"],
            train["grid"],
            train["potential"],
        )
        train_acc = accuracy(model, train)
        val_acc = accuracy(model, val)

        # if (epoch % (trainer_config["epochs"] / 100)) == 0:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
