import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import hypermagnetics.sources as sources
import wandb
from hypermagnetics.models import HyperMLP

sweep_configuration = {
    "method": "grid",
    "name": "grid-search",
    "parameters": {
        "width": {"values": [16, 32, 64]},
        "depth": {"values": [2, 3, 4]},
        "hdepth": {"values": [1, 2, 3]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="hypermagnetics")


def loss(model, sources, grid, target):
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    return jnp.mean(optax.huber_loss(pred, target))


@eqx.filter_jit
def step(model, optim, opt_state, sources, grid, target):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, sources, grid, target)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optim, opt_state, loss_value


@eqx.filter_jit
def accuracy(model, data):
    sources, grid, target = data["sources"], data["grid"], data["potential"]
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    diff = jnp.linalg.norm(target - pred)
    return jnp.median(diff / jnp.linalg.norm(target) * 100)


def main():
    run = wandb.init()
    width = wandb.config.width
    depth = wandb.config.depth
    hdepth = wandb.config.hdepth

    source_config = {"N": 500, "M": 3, "lim": 3, "res": 32}
    train = sources.configure(**source_config, key=jr.PRNGKey(40))
    val = sources.configure(**source_config, key=jr.PRNGKey(41))

    hyperkey, mainkey = jr.split(jr.PRNGKey(42), 2)
    model_config = {
        "width": width,
        "depth": depth,
        "hdepth": hdepth,
    }
    model = HyperMLP(**model_config, hyperkey=hyperkey, mainkey=mainkey)

    trainer_config = {"learning_rate": 0.001, "epochs": 10000}
    optim = optax.adam(trainer_config["learning_rate"])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for epoch in range(trainer_config["epochs"]):
        model, optim, opt_state, train_loss = step(
            model,
            optim,
            opt_state,
            train["sources"],
            train["grid"],
            train["potential"],
        )
        train_acc = accuracy(model, train)
        val_acc = accuracy(model, val)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )


# Start sweep job.
wandb.agent(sweep_id, function=main)
