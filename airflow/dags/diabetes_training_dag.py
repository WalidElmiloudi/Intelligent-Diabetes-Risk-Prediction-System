from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.append("/opt/airflow")

from src.data.preprocessing import preprocess
from src.clustering.cluster_analysis import cluster
from src.classification.tuning import tune




def preprocess_data():
    print("Preprocessing the data...")
    preprocess()


def clustering():
    print("Clustering and Analyzing clusters ...")
    cluster()


def fine_tuning():
    print("Fine tuning the best model's hyperparameters...")
    tune()


default_args = {
    "owner": "Diabetes-Risk-project",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="Diabetes_risk_pipeline",
    default_args=default_args,
    description="Predicting the patient's Diabetes Risk Based on some Patient informations",
    schedule="0 0 10 * *",
    start_date=datetime(2026, 10, 10),
    catchup=False,
    tags=["Diabetes"],
) as dag:

    preprocessing = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    clustering_data = PythonOperator(
        task_id="clustering_data",
        python_callable=clustering,
    )

    tuning = PythonOperator(
        task_id="hyperparameters_optimization",
        python_callable=fine_tuning,
    )

    preprocessing >> clustering_data >> tuning