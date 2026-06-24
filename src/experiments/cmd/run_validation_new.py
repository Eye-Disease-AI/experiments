"""
Description of what this entrypoint should achieve:
It should perform some kind of verification. I can see that
there are two modes:

* seeds with --num-seeds option
* kfold with -K option

It creates separete validation study with the first component
matching the original name. Study is run as part of the same
MLFlow experiment.

1. We load best study params
2. Set up data modules

In the legacy code we set up Nuclear Cataract datamodule using
the params, but now we are leaving decision about any needed modules to the
study class. So either study class should expose publicly method for getting its
datamodules OR we can consturct them ourselves based on the config OR we can
move all validation logic to be part of study class.

3. MLFlow orchestration
4. Look for parameters with `best_` prefix in the selected study, start a run
and log these values there
5. Start nested run named `retrain` and do the actual training there using earlier
loaded and logged best parameters
6. Run either kfold or seeds validation based on CLI options. These modes also
start nested runs and log parameters which are mode-specific and run names
contain information about which seed or which fold was used
"""
