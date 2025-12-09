import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from neuralop.models import FNO
from neuralop.training import Trainer, AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from pathlib import Path
import warnings
import wandb

warnings.filterwarnings("ignore", category=UserWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"


class PotentialDataset(torch.utils.data.Dataset):
    def __init__(self, run_name, cfg, n_samples=1000, n_sources=1, seed=42, val=False):
        self.datapath = Path(__file__).parent / ".." / ".." / "data"
        self.cfg = cfg
        if val:
            self.db_name = f"val_{run_name}_{seed}_{n_samples}_{n_sources}_fno.h5"
        else:
            self.db_name = f"train_{run_name}_{seed}_{n_samples}_{n_sources}_fno.h5"
        self.size = n_samples

    def open_hdf5(self):
        self.db = h5py.File(self.datapath / self.db_name, "r")

    def __getitem__(self, idx):
        if not hasattr(self, "db"):
            self.open_hdf5()
        # Shape: CxHxW
        input = (
            np.array(self.db["input"][idx])
            .reshape(self.cfg["res"], self.cfg["res"], -1)
            .transpose(2, 0, 1)
        )
        output = np.array(self.db["output"][idx]).reshape(
            self.cfg["res"], self.cfg["res"]
        )[None, ...]  # add channel dim
        return {
            "x": torch.from_numpy(input.astype("float32")),
            "y": torch.from_numpy(output.astype("float32")),
        }

    def __len__(self):
        return self.size


if __name__ == "__main__":
    config = {
        "shape": "prism",
        "n_samples": 200000,
        "lim": 1.2,
        "res": 32,
        "dim": 2,
        "epochs": 100,
        "seed": 42,
        "lambda_field": 0.25,
        "batch_size": 2500,
        "hidden_channels": 64,
        "n_modes": 10,
    }

    # Set up WandB logging
    wandb.login(key="778222655164cfc7dc659dd0c13d42d02766649c")
    wandb_args = dict(
        config=config,
        project="hypermagnetics",
        entity="dl4mag",
    )
    wandb.init(**wandb_args)

    train_dataset = PotentialDataset(
        run_name="res32_large_m",
        cfg=config,
        n_samples=config["n_samples"],
        n_sources=1,
        seed=config["seed"],
    )

    val_dataset = PotentialDataset(
        run_name="res32_large_m",
        cfg=config,
        n_samples=1000,
        n_sources=10,
        seed=41,
        val=True,
    )

    val_single_dataset = PotentialDataset(
        run_name="res32_large_m",
        cfg=config,
        n_samples=1000,
        n_sources=1,
        seed=40,
        val=True,
    )

    operator = FNO(n_modes=(10, 10), hidden_channels=64, in_channels=2, out_channels=1)
    operator = operator.to(device)
    n_params = count_model_params(operator)
    print(f"\nOur model has {n_params} parameters.")

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=48,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(1000, config["batch_size"]),
        shuffle=False,
        num_workers=48,
        pin_memory=True,
        persistent_workers=False,
    )

    val_single_loader = DataLoader(
        val_single_dataset,
        batch_size=min(1000, config["batch_size"]),
        shuffle=False,
        num_workers=48,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {"val": val_loader, "val_single": val_single_loader}

    optimizer = AdamW(operator.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    l1loss = LpLoss(d=1, p=1)  # L2 loss for function values
    h1loss = H1Loss(d=1)  # H1 loss includes gradient information

    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l1loss}

    # Create the trainer
    trainer = Trainer(
        model=operator,
        n_epochs=config["epochs"],
        device=device,
        data_processor=DefaultDataProcessor().to(device),
        wandb_log=True,  # Disable Weights & Biases logging for this tutorial
        eval_interval=1,  # Evaluate every 5 epochs
        use_distributed=False,  # Single GPU/CPU training
        verbose=True,
    )

    # train the model
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    model_folder = Path(__file__).resolve() / ".." / ".." / "models" / "fno"
    operator.save_checkpoint(save_folder=model_folder, save_name="fno_large_m")
