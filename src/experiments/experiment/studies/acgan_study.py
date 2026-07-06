from dataclasses import dataclass, replace
from typing import Any

import lightning as L
import mlflow
import torch
from dataset.hard_policy import HardPolicy
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import override

from experiments.experiment.datamodules.datamodule import DataModule, DataModuleConfig
from experiments.experiment.datamodules.nuclear_cataract_datamodule import (
    NuclearCataractDataModule,
    NuclearCataractDataModuleConfig,
)
from experiments.experiment.models.acgan import ACGANModule, ACGANModuleConfig
from experiments.experiment.models.convnext import ConvNext, ConvNextConfig
from experiments.experiment.studies.study import Study, StudyConfig
from experiments.lib.config_serializing import OptunaOptimised, resolved
from experiments.lib.reproducibility import global_seed_rng


@dataclass(frozen=True, kw_only=True)
class ACGANStudyConfig(StudyConfig):
    gen_datamodule_config: DataModuleConfig = NuclearCataractDataModuleConfig(
        batch_size=8,
        return_paths=False,
        cache=True,
        hard_policy=HardPolicy.PASSTHROUGH,
        image_size=224,
        normalize=False,
        augment_rot_angle=15,
    )
    clf_datamodule_config: DataModuleConfig = NuclearCataractDataModuleConfig(
        batch_size=OptunaOptimised("categorical", {"choices": [64]}),
        return_paths=False,
        cache=True,
        hard_policy=HardPolicy.PASSTHROUGH,
        image_size=224,
        normalize=True,
        augment_rot_angle=15,
    )
    clf_model_config: ConvNextConfig = ConvNextConfig(
        learning_rate=1e-4,
        weight_decay=1e-8,
        dropout=0.3,
    )
    gen_model_config: ACGANModuleConfig = ACGANModuleConfig(
        # I think we should support some way of saying a kw is undefined!
        n_classes=-1,
        class_names=[],
        latent_dim=OptunaOptimised("int", {"low": 50, "high": 200, "log": True}),
        img_size=224,
        num_channels=3,
    )
    max_clf_epochs: int = 100
    max_gen_epochs: int = 100
    use_early_stopping: bool = True
    early_stopping_patience: int = 5

    @override
    @staticmethod
    def get_configured_class():
        return ACGANStudy


class ACGANStudy(Study):
    _config: ACGANStudyConfig

    def __init__(self, config: ACGANStudyConfig):
        super().__init__(config)
        self._config = config

    @override
    def _init_datamodules(self) -> list[DataModule]:
        self._gen_datamodule: NuclearCataractDataModule = (
            self._config.gen_datamodule_config.build()
        )
        return [self._gen_datamodule]

    @override
    def _configure_datamodules(self, params: dict[str, Any]) -> None:
        # Nothing to configure
        pass

    def _bake_clf_model_config(self, config: ACGANStudyConfig) -> ConvNextConfig:
        return replace(
            config.clf_model_config,
            class_weights=self._clf_datamodule.class_weights,
            n_classes=self._clf_datamodule.n_classes,
        )

    def _bake_gen_model_config(self, config: ACGANStudyConfig) -> ACGANModuleConfig:
        return replace(
            config.gen_model_config,
            n_classes=self._gen_datamodule.n_classes,
            class_names=self._gen_datamodule.class_names,
        )

    def _build_augmented_clf_datamodule(
        self, config: ACGANStudyConfig, fake_dataset: Dataset
    ) -> NuclearCataractDataModule:
        self._clf_datamodule: NuclearCataractDataModule = (
            config.clf_datamodule_config.build()
        )
        self._clf_datamodule.setup(stage="fit")
        self._clf_datamodule.setup_augment(fake_dataset)
        return self._clf_datamodule

    def _create_clf_trainer(
        self,
        logger,
        max_epochs=None,
        callbacks=None,
    ):
        callbacks = [] if callbacks is None else callbacks
        if self._config.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self._config.early_stopping_patience,
                    mode="min",
                )
            )

        return L.Trainer(
            max_epochs=(max_epochs or self._config.max_clf_epochs),
            accelerator=self._config.device,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            precision=self._config.gpu_precision,  # type: ignore
            deterministic=True,
        )

    def _create_gen_trainer(
        self,
        logger,
    ):
        return L.Trainer(
            max_epochs=self._config.max_gen_epochs,
            accelerator=self._config.device,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            precision=self._config.gpu_precision,  # type: ignore
            deterministic=True,
        )

    def _create_fake_dataset(self, gen_module: ACGANModule):
        num_samples = 100
        fake_imgs = []
        fake_labels = []

        gen_module.eval()
        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                sample_noise = torch.randn(
                    (1, resolved(gen_module.config.latent_dim)),
                )
                sample_labels = torch.randint(
                    0, self._gen_datamodule.n_classes, (1,)
                ).to(dtype=torch.long)
                fake_img = gen_module(sample_noise, sample_labels).squeeze()
                fake_imgs.append(fake_img)
                fake_labels.append(sample_labels.squeeze())

        return FakeDataset(fake_imgs, fake_labels)

    @override
    def _train(self, params: dict[str, Any], get_logger, callbacks) -> None:
        config: ACGANStudyConfig = self._set_config_optuna_params(params)
        gen_module: ACGANModule = self._bake_gen_model_config(config).build()
        gen_logger: MLFlowLogger = get_logger(config.gen_model_config.class_name())
        gen_trainer = self._create_gen_trainer(logger=gen_logger)

        global_seed_rng(config.seed)
        gen_trainer.fit(gen_module, datamodule=self._gen_datamodule)

        print("Sampling from trained GAN to create augmented dataset")
        fake_dataset = self._create_fake_dataset(gen_module)

        global_seed_rng(config.seed)
        clf_datamodule = self._build_augmented_clf_datamodule(config, fake_dataset)

        clf_module: ConvNext = self._bake_clf_model_config(config).build()
        clf_logger: MLFlowLogger = get_logger(config.clf_model_config.class_name())
        # TODO: We pass callbacks only here, should it be like that?
        clf_trainer = self._create_clf_trainer(logger=clf_logger, callbacks=callbacks)

        global_seed_rng(config.seed)
        clf_trainer.fit(clf_module, datamodule=clf_datamodule)

    @override
    def _retrain(
        self, get_logger, best_params: dict, best_epoch: int, callbacks: list
    ) -> tuple[L.Trainer, Any]:
        global_seed_rng(self._config.seed)
        self._init_datamodules()
        config = self._set_config_optuna_params(best_params)
        self._gen_datamodule.batch_size = config.gen_datamodule_config.batch_size
        self._gen_datamodule.setup(stage="fit")

        best_gen_module: ACGANModule = self._bake_gen_model_config(config).build()
        gen_trainer = self._create_gen_trainer(
            logger=get_logger(self._config.gen_model_config.class_name())
        )
        global_seed_rng(config.seed)
        gen_trainer.fit(best_gen_module, datamodule=self._gen_datamodule)

        fake_dataset = self._create_fake_dataset(best_gen_module)

        global_seed_rng(config.seed)
        clf_datamodule = self._build_augmented_clf_datamodule(config, fake_dataset)

        best_clf_model: ConvNext = self._bake_clf_model_config(config).build()
        clf_trainer = self._create_clf_trainer(
            max_epochs=best_epoch + 1,
            callbacks=callbacks,
            logger=get_logger(self._config.clf_model_config.class_name()),
        )
        global_seed_rng(config.seed)
        clf_trainer.fit(
            best_clf_model,
            datamodule=clf_datamodule,
        )

        validation_metrics = clf_trainer.validate(
            best_clf_model, datamodule=clf_datamodule
        )
        retrain_metric = validation_metrics[0][self._config.optuna_metric]
        mlflow.log_metric(f"retrain_{self._config.optuna_metric}", retrain_metric)

        # TODO: No way to pass gan trainer and save its weights!
        return clf_trainer, validation_metrics


class FakeDataset(Dataset):
    def __init__(self, images: list[torch.Tensor], labels: list[torch.Tensor]):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]
