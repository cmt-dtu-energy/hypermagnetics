import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from hypermagnetics.hypermlp import HyperMLP
from hypermagnetics.sources import Sources

optim = optax.adam(learning_rate=0.01)


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
    sources, grid, target = data.m_r, data.grid, data.potential
    pred = jax.vmap(model, in_axes=(0, None))(sources, grid)
    diff = jnp.linalg.norm(target - pred)
    return jnp.median(diff / jnp.linalg.norm(target) * 100)


if __name__ == "__main__":
    train_sources = Sources(N=100, M=2, key=jr.PRNGKey(40), lim=3, res=32)
    test_sources = Sources(N=100, M=2, key=jr.PRNGKey(41), lim=3, res=32)

    steps = 10_000
    logger = {"train_loss": [], "train_acc": [], "val_acc": []}
    model = HyperMLP(16, 3, *jr.split(jr.PRNGKey(42), 2))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for step in range(steps):
        model, opt_state, train_loss = step(model, opt_state)

        if (step % (steps / 100)) == 0:
            logger["train_loss"].append(train_loss)
            logger["train_acc"].append(accuracy(model, train_sources))
            logger["val_acc"].append(accuracy(model, test_sources))

        if (step % (steps / 100)) == 0:
            print(
                f"{step=}, train_loss={train_loss:.4f}, ",
                f"accuracy={logger['train_acc'][-1]:.4f}",
                f"val_accuracy={logger['val_acc'][-1]:.4f}",
            )
