import equinox as eqx
import jax.random as jr
import optax
import yaml

import hypermagnetics.sources as sources
import wandb
from hypermagnetics import plots
from hypermagnetics.measures import accuracy, loss

# from hypermagnetics.models import AdditiveMLP, HyperMLP, save  # noqa: F401
from hypermagnetics.models.hyper_fourier import FourierModel


def fit(trainer_config, optim, model, train, val, log=print, every=1):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, data):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, data)
        updates, opt_state = optim.update(grads, opt_state, model)

        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(trainer_config["epochs"]):
        model, opt_state, train_loss = step(model, opt_state, train)
        train_err = accuracy(model, train)
        val_err = accuracy(model, val)
        log(
            {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "train_err": train_err.item(),
                "val_err": val_err.item(),
            }
        ) if (epoch % every == 0) else None
        if train_err < 1.0:
            break

    return model


if __name__ == "__main__":
    with open("config/run-configuration.yaml", "r") as f:
        run_configuration = yaml.safe_load(f)

    source_config = run_configuration["source"]
    train = sources.configure(**source_config, key=jr.PRNGKey(100))
    val = sources.configure(**source_config, key=jr.PRNGKey(101))

    key = jr.PRNGKey(42)
    run_configuration["model"] = {"order": 64}  # Hijack run configuration
    model_config = run_configuration["model"]
    # model = HyperMLP(**model_config, key=key)
    # model = AdditiveMLP(**model_config, key=key)
    model = FourierModel(**model_config, key=key)

    schedule = run_configuration["schedule"]
    wandb.init(
        entity="dl4mag",
        project="hypermagnetics",
        config=run_configuration,
    )
    wandb.log({"nparams": model.nparams})

    for trainer_config in schedule:
        optim = optax.adam(**trainer_config["params"])
        model = fit(trainer_config, optim, model, train, val, log=wandb.log, every=10)

    # save(model, wandb.run.id)
    plots(train, model, idx=0, output="wandb")
    wandb.finish()
