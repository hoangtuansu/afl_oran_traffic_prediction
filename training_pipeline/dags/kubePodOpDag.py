from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime, timedelta

from kubernetes.client import models as k8s

default_args = {
    'owner': 'taisp',
    'depends_on_past': False,
    'start_date': datetime(2020, 1, 1),
    'retries': 7,
    'retry_delay': timedelta(minutes=5),
    'namespace': 'taisp-ml-infra',
    'in_cluster': True,  # if set to true, will look in the cluster, if false, looks for file
    'get_logs': True,
    'is_delete_operator_pod': True
}

dag = DAG('fl_training',
          default_args=default_args,
          description='Federated Training Dag',
          schedule_interval='0 12 * * *',
          start_date=datetime(2017, 3, 20),
          catchup=False)

env_var = [k8s.V1EnvVar(name='MLFLOW_TRACKING_URI', value='http://10.180.113.115:32256'), 
           k8s.V1EnvVar(name='MLFLOW_TRACKING_USERNAME', value='user'),
           k8s.V1EnvVar(name='MLFLOW_TRACKING_PASSWORD', value='sr9TvkIjaj')]]

fl_server = KubernetesPodOperator(
            image="hoangtuansu/fl_server:0.1",
            env_vars=env_var,
            name=f"server",
            task_id=f"server_training",
            retries=5,
            retry_delay=timedelta(minutes=5),
            dag=dag,
        )

fl_client = KubernetesPodOperator(
            image="hoangtuansu/fl_client:0.1",
            env_vars=env_var,
            name=f"client",
            task_id=f"client_training",
            retries=5,
            retry_delay=timedelta(minutes=5),
            dag=dag,
        )


ingest_data >> load_data