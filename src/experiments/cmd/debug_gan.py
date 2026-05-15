import lightning as L

from experiments.experiment.data import MyDataModule
from experiments.experiment.models.gan import GAN
from experiments.lib.reproducibility import RNG

if __name__ == "__main__":
    rng = RNG()
    rng.set_seed(42)
    datamodule = MyDataModule(rng, cache=False, num_workers=3)
    datamodule.setup()
    datamodule.prepare_data()
    model = GAN(datamodule.dataset.n_classes)

    trainer_params = {
        "max_epochs": 100,
        "accelerator": "auto",
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        "log_every_n_steps": 1,
        "deterministic": True,
    }

    trainer = L.Trainer(**trainer_params)
    trainer.fit(model, datamodule)
