import logging

# Disable annoying and harmless logs
import warnings
from optuna.exceptions import ExperimentalWarning
def stop_logs():
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    class _TipFilter(logging.Filter):
        def filter(self, record):
            return "💡 Tip" not in record.getMessage()
    tip_filter = _TipFilter()
    for name in ("lightning.pytorch", ""):
        for handler in logging.getLogger(name).handlers:
            handler.addFilter(tip_filter)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
