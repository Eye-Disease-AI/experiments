from experiments.experiment.studies.cam_study import CAMStudyConfig


def main():
    cam_study_config = CAMStudyConfig.from_classifier_study(
        "augmentations",
        "augmentations_R_6ea4ecb",
    )
    cam_study = cam_study_config.build()
    cam_study.run()


if __name__ == "__main__":
    main()
