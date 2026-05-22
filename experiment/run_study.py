import matplotlib.pyplot as plt
import mlflow
import optuna
from experiment.data import MyDataModule
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
from experiment import common_config 
CONFIG_PARAMS = {
    k: str(v) for k, v in vars(common_config).items()
    if k.isupper() and not k.startswith("_")
}
from experiment.objective import create_trainer, objective
from lib.mlflow_setup import Experiment
from lib.reproducibility import RNG


def run_study(
    ModelClass,
    study_name: str,
    exp: Experiment,
    rng: RNG,
    datamodule: MyDataModule,
    retrain = True,
):
    study = optuna.create_study(
        study_name=study_name,
        direction={"min": "minimize", "max": "maximize"}[common_config.OPTUNA_DIRECTION],
        storage=exp.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    exp.study = study
    study.set_user_attr("model_class", ModelClass.__name__)
    study.set_user_attr("config", CONFIG_PARAMS)
    study.set_user_attr("hard_policy", datamodule.hard_policy.name)

    n_finished = sum(
        1 for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
    )
    print(f"Study '{study_name}': {n_finished}/{common_config.MAX_TRIALS} finished trials.")

    parent_run_id = study.user_attrs.get("mlflow_parent_run_id")
    if parent_run_id:
        parent_run = mlflow.start_run(run_id=parent_run_id)
    else:
        parent_run = mlflow.start_run(run_name=study_name)
        study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)
        mlflow.set_tag("optuna_study", study_name)
        mlflow.set_tag("model", ModelClass.__name__)
        mlflow.log_params(CONFIG_PARAMS)
        mlflow.log_param("hard_policy", datamodule.hard_policy.name)

    if n_finished >= common_config.MAX_TRIALS:
        print(
            f"Study already complete. Best: {study.best_params}, {common_config.OPTUNA_METRIC}: {study.best_value:.4f}"
        )
        mlflow.end_run()
        return

    study.optimize(
        lambda trial: objective(datamodule, rng, exp, trial, ModelClass),  # pyright: ignore
        n_trials=common_config.MAX_TRIALS * 2,
        callbacks=[
            MaxTrialsCallback(
                common_config.MAX_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED)
            )
        ],
    )

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print("No completed trials. Skipping retrain and plots.")
        mlflow.end_run()
        return

    try:
        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metric(f"best_{OPTUNA_METRIC}", best_trial.value)  # pyright: ignore
        mlflow.set_tag("best_trial_number", best_trial.number)
        agg = min if common_config.OPTUNA_DIRECTION == "min" else max
        best_epoch = agg( # pyright: ignore
            best_trial.intermediate_values, key=best_trial.intermediate_values.get # pyright: ignore
        )
        mlflow.log_metric("best_epoch", best_epoch)

        best_params = study.best_params
        rng.set_seed(common_config.SEED)
        datamodule.batch_size = best_params["batch_size"]  # pyright: ignore
        datamodule.setup(stage="fit")
        model_params = {k: v for k, v in best_params.items() if k != "batch_size"}
        class_weights = datamodule.train_class_weights  # pyright: ignore
        best_model = ModelClass(
            datamodule.dataset.n_classes, **model_params, class_weights=class_weights
        )  # pyright: ignore

        if retrain:
            best_model_path = f"checkpoints/{study_name}/best.ckpt"
            trainer = create_trainer(
                run=parent_run,
                max_epochs=best_epoch+1,
                callbacks=[]
            )
            trainer.fit(best_model, datamodule=datamodule)
            trainer.save_checkpoint(best_model_path)
            out = trainer.validate(best_model, datamodule=datamodule)
            retrain_metric = out[0][common_config.OPTUNA_METRIC]
            mlflow.log_metric(f"retrain_{common_config.OPTUNA_METRIC}", retrain_metric)
            mlflow.log_artifact(best_model_path, artifact_path="model")

        try:
            fig = plot_optimization_history(study)
            mlflow.log_figure(fig.figure, "optimization_history.png")
            plt.close()
        except Exception:
            pass

        try:
            fig = plot_param_importances(study)
            mlflow.log_figure(fig.figure, "param_importances.png")
            plt.close()
        except Exception:
            pass
    finally:
        mlflow.end_run()
