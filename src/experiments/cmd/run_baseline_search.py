from experiments.experiment.studies.baseline_search_study import (
    BaselineSearchStudyConfig,
)

def main():
    baseline_study_config = BaselineSearchStudyConfig(
        experiment_name="baseline_search",
    )
    baseline_study = baseline_study_config.build()

    print("Running baseline search study...")
    baseline_study.run()


if __name__ == "__main__":
    main()
