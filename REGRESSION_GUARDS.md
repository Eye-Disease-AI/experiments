# Regression guards

List of manual tests to perform before merging a PR to make sure the changes
do not introduce regression:

1. Run all of the scripts in `src/experiments/cmd` and make sure that
every of these files finish without errors. For longer studies you can
manually set low number of `max_epochs` and `num_trials` to make them finish
quickly.
2. Make sure that `run_ci_poc.py` does not return any NaN values in
the results - this may indicate that values are logged incorrectly.
3. Make sure that in a validation run e.g. `run_baseline.py` metrics
logged are such that `best_*` metrics from parent run are matching `retrain_`
metrics from `retrain` child run.
4. When running `run_validation.py` make sure to test both `kfold` mode
and `seeds` mode. For `kfold` pass at least 2 folds and for seeds - at least
2 seeds.
5. Make sure that for seeds child runs the metrics values differ between
child runs. If it is not the case it may indicate that seeding is broken.
6. Make sure that for kfold child runs the metrics values differ between
child runs. If it is not the case it may mean a study incorrectly generates
dataset folds or fold validation logic is broken in the study.
7. In the "regular" run make sure that best logged metric (it is only one -
the optimized metric) is the same as corresponding `retrain_` metric. If
it is not the case it indicates either seeding issue or that model is
retrained in a different way compared to the original training.
8. For the "regular" run make sure that mertics logged in the parent run
are the same as one of the trials metrics.
