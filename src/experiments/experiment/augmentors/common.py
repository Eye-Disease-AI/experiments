import matplotlib.pyplot as plt
import mlflow
import mlflow.entities
import pandas
import torch

from experiments.experiment.augmentors.augmentor import Augmentor, AugmentorConfig
from experiments.lib.mlflow_setup import Experiment


def create_augmentor(config: str) -> AugmentorConfig:
    from experiments.experiment.augmentors.stylegan_augmentor import (
        StyleganAugmentorConfig,
    )
    from experiments.experiment.augmentors.tacgan_augmentor import (
        TacganAugmentorConfig,
    )

    if config == "stylegan-r":
        return StyleganAugmentorConfig.known_config_r()
    if config == "stylegan-t":
        return StyleganAugmentorConfig.known_config_t()
    if config == "tacgan-ac":
        return TacganAugmentorConfig.known_config_ac()
    if config == "tacgan-tac1":
        return TacganAugmentorConfig.known_config_tac1()
    if config == "tacgan-tac2":
        return TacganAugmentorConfig.known_config_tac2()
    if config == "tacgan-tac3":
        return TacganAugmentorConfig.known_config_tac3()
    raise RuntimeError("unknown config")


def query_params_filter_string(query_params) -> str:
    param_filters = []

    for param in query_params:
        if param[1] in ["True", "False"]:
            raise RuntimeError("Filter string does not support bool filtering")
        param_filters.append("params.'" + param[0] + "' = \"" + param[1] + '"')

    filter_string = " and ".join(param_filters)
    filter_string += "and attribute.status = 'FINISHED'"

    return filter_string


def find_run(
    query_params: list[tuple[str, str]],
    exp: Experiment,
    fail_if_not_exists: bool = False,
) -> mlflow.entities.Run | None:
    non_bool_params = []
    bool_params = []

    for param in query_params:
        if param[1] in ("True", "False"):
            bool_params.append(param)
        else:
            non_bool_params.append(param)

    filter_string = query_params_filter_string(non_bool_params)
    # mlflow paginates internally and we don't have to it ourselves
    runs = mlflow.search_runs(
        experiment_ids=[exp.mlflow_experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
    )
    assert isinstance(runs, pandas.DataFrame)

    for bool_name, bool_val in bool_params:
        col_name = f"params.{bool_name}"

        if bool_val == "True":
            filtered_runs = runs[runs[col_name] == bool_val]
        elif bool_val == "False":
            filtered_runs = runs[(runs[col_name] == bool_val) | runs[col_name].isna()]

    if len(bool_params) == 0:
        filtered_runs = runs

    filtered_runs = filtered_runs[:1]

    if not fail_if_not_exists and len(filtered_runs) == 0:
        return None

    assert len(filtered_runs) == 1, (
        f"Training run not found or ambiguous result (found {len(filtered_runs)})"
    )

    run_id = filtered_runs["run_id"].iloc[0]  # type: ignore
    return exp.client.get_run(run_id)


def sample_augmentor(augmentor: Augmentor, save_path_prefix: str):
    seeds = [3, 4]
    classes = [0, 1]
    gens = augmentor.generate(
        torch.Tensor(seeds), torch.Tensor(classes).to(dtype=torch.long)
    )
    for i, gen in enumerate(gens):
        gen_perm = gen.permute(1, 2, 0)
        plt.imshow(gen_perm)
        plt.savefig(f"{save_path_prefix}_seed_{seeds[i]}_class{classes[i]}.png")
