import equinox as eqx
import jax.random as jr
import optax

import hypermagnetics.sources as sources
from hypermagnetics.measures import accuracy, loss
from hypermagnetics.models import HyperMLP


def fit(trainer_config, optim, model, train, val, log=print):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, data):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, data)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(trainer_config["epochs"]):
        model, opt_state, train_loss = step(model, opt_state, train)
        train_acc = accuracy(model, train)
        val_acc = accuracy(model, val)
        log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

    step._clear_cache()


if __name__ == "__main__":
    source_config = {
        "n_samples": 10,
        "n_sources": 2,
        "lim": 3,
        "res": 32,
    }
    train = sources.configure(**source_config, key=jr.PRNGKey(40))
    val = sources.configure(**source_config, key=jr.PRNGKey(41))

    hyperkey, mainkey = jr.split(jr.PRNGKey(42), 2)
    model = HyperMLP(16, 3, 1, 2, hyperkey=hyperkey, mainkey=mainkey)

    trainer_config = {
        "learning_rate": 1e-5,
        "momentum": 0.99,
        "epochs": 1000,
    }
    optim = optax.adam(
        learning_rate=trainer_config["learning_rate"],
        # momentum=trainer_config["momentum"],
        # nesterov=True,
    )
    fit(trainer_config, optim, model, train, val)
