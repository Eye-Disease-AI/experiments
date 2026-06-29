import os
from urllib.parse import quote_plus

import mlflow
import optuna
import psycopg2
from dotenv import load_dotenv
from optuna.storages import RDBStorage


class Experiment:
    def __init__(self, experiment_name: str | None = None):
        self.experiment_name = experiment_name
        self.study: optuna.study.Study | None = None

        load_dotenv()
        self.init_server_data()

        try:
            print(f"Connecting to MLFLOW server {self.MLFLOW_URI}")
            self.connect()
        except Exception as e:
            print(f"Failed to conect to the MLFLow server {self.MLFLOW_URI}")
            print(e)
            exit(1)
        print("Connection OK")

    def init_server_data(self):
        self.MLFLOW_TRACKING_USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"] = (
            os.environ["MLFLOW_ADMIN_USERNAME"]
        )
        self.MLFLOW_TRACKING_PASSWORD = os.environ["MLFLOW_TRACKING_PASSWORD"] = (
            os.environ["MLFLOW_ADMIN_PASSWORD"]
        )
        self.DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
        self.DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
        self.DB_USER = os.environ["POSTGRES_USER"]
        self.DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]
        self.DB_NAME = os.environ["POSTGRES_DB"]
        encoded_password = quote_plus(self.DB_PASSWORD)
        self.OPTUNA_DB_URL = f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/optuna"
        self.MLFLOW_URI = f"http://{os.environ.get('MLFLOW_IP', 'localhost')}:{os.environ.get('MLFLOW_PORT', '5000')}"
        self.experiment_name = self.experiment_name

    def connect(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)

        if self.experiment_name:
            self.mlflow_experiment = mlflow.set_experiment(self.experiment_name)

        self.client = mlflow.MlflowClient()
        self.storage = RDBStorage(url=self.OPTUNA_DB_URL)

        conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            dbname=self.DB_NAME,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'optuna'")
            if not cur.fetchone():
                cur.execute("CREATE DATABASE optuna")
                print("Created database 'optuna'.")
        conn.close()
