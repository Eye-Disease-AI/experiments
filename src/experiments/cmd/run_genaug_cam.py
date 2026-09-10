import argparse
from dataclasses import replace

from experiments.experiment.studies.cam_study import CAMStudyConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cam_study_config = CAMStudyConfig.from_classifier_study(
        args.experiment_name,
        args.run_name,
    )
    cam_study_config = replace(cam_study_config, study_suffix=args.run_name)
    cam_study = cam_study_config.build()
    cam_study.run()


if __name__ == "__main__":
    main()
