from experiments.experiment.studies.acgan_study import ACGANStudy, ACGANStudyConfig

study = ACGANStudy(
    ACGANStudyConfig(
        experiment_name="acgan-augmentation",
        # device="cpu",
        max_gen_epochs=10,
        max_clf_epochs=1,
        gpu_precision="32",
    )
)
study.run()
