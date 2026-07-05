from experiments.experiment.studies.acgan_study import ACGANStudy, ACGANStudyConfig

study = ACGANStudy(
    ACGANStudyConfig(
        experiment_name="acgan-augmentation4",
        # device="cpu",
        max_gen_epochs=2,
        max_clf_epochs=3,
        gpu_precision="32",
        max_trials=2,
        # optuna_metric="ConvNext-val_auroc",
    )
)
study.run()
