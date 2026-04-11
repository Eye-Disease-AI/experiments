import warnings
import torch
import mlflow
import psycopg2
import os
from dotenv import load_dotenv
from optuna.storages import RDBStorage

class ServerConnection():
    def __init__(self, experiment_name):
        self.MLFLOW_TRACKING_USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_ADMIN_USERNAME"]
        self.MLFLOW_TRACKING_PASSWORD =  os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_ADMIN_PASSWORD"]
        self.DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
        self.DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
        self.DB_USER = os.environ["POSTGRES_USER"]
        self.DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]
        self.DB_NAME = os.environ["POSTGRES_DB"]
        self.OPTUNA_DB_URL = f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/optuna"
        self.MLFLOW_URI = "http://localhost:5000"
        self.experiment_name = experiment_name
        self.client = None
        self.storage = None

    def connect(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        mlflow.set_experiment(self.experiment_name)

        self.client = mlflow.MlflowClient()
        self.storage = RDBStorage(url=self.OPTUNA_DB_URL)
        
        conn = psycopg2.connect(host=self.DB_HOST, port=self.DB_PORT, user=self.DB_USER, password=self.DB_PASSWORD, dbname=self.DB_NAME)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'optuna'")
            if not cur.fetchone():
                cur.execute("CREATE DATABASE optuna")
                print("Created database 'optuna'.")
        conn.close()

    def save_model(self, model, example_input=None):
        def _save_as_torchscript():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.pytorch")
                mlflow.pytorch.log_model(
                    torch.jit.script(model),
                    name="best_model",
                    serialization_format="pickle",
                )
            mlflow.set_tag("model_serialization_format", "torchscript_pickle")

        if example_input is None:
            _save_as_torchscript()
            return

        try:
            mlflow.pytorch.log_model(
                model,
                name="best_model",
                serialization_format="pt2",
                input_example=example_input,
            )
            mlflow.set_tag("model_serialization_format", "pt2")
        except Exception as e:
            _save_as_torchscript()

class Experiment():
    def __init__(self, experiment_name):
        load_dotenv()
        self.srv = ServerConnection(experiment_name)
        self.srv.connect()
        self.study = None

    def set_study(self, study):
        self.study = study
