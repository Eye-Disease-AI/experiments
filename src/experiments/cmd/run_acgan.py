from experiments.experiment.studies.acgan_study import ACGANStudy, ACGANStudyConfig

study = ACGANStudy(ACGANStudyConfig(device="cpu"))
study.run()
