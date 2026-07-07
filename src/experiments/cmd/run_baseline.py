from experiments.experiment.studies.baseline_study import BaselineStudyConfig

def main():
    baseline_study_config = BaselineStudyConfig(
        experiment_name="baseline",
    )
    baseline_study = baseline_study_config.build()

    print("Running baseline search study...")
    baseline_study.run()


if __name__ == "__main__":
    main()
