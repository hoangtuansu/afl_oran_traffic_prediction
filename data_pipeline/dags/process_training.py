from airflow import DAG
import os
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
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

dag = DAG('xapp_ad',
          default_args=default_args,
          description='Anormaly Dection xApp Dag',
          schedule=None,
          start_date=datetime(2017, 3, 20),
          catchup=False)

COMPONENT_IMAGE_VERSION = os.getenv('COMPONENT_IMAGE_VERSION')

ingest_data = KubernetesPodOperator(
            image=f"hoangtuansu/ingester:{COMPONENT_IMAGE_VERSION}",
            env_vars=[k8s.V1EnvVar(name='RAW_DATA_TABLE', value='RawPM')],
            image_pull_policy="Always",
            name=f"ingest_data",
            task_id=f"ingest_data",
            cmds=["bash", "-c"],
            arguments=["python ingest_data.py"],
            on_finish_action='keep_pod',
            dag=dag,
        )


clean_data = KubernetesPodOperator(
            image=f"hoangtuansu/cleaner:{COMPONENT_IMAGE_VERSION}",
            env_vars=[
                    k8s.V1EnvVar(name='RAW_DATA_TABLE', value='RawPM'),
                    k8s.V1EnvVar(name='CLEAN_TRAIN_DATA_TABLE', value='TrainPM'),
                    k8s.V1EnvVar(name='CLEAN_TEST_DATA_TABLE', value='TestPM'),
                    k8s.V1EnvVar(name='CLEAN_ACTUAL_DATA_TABLE', value='ActualPM')
                    ],
            name=f"clean_data",
            image_pull_policy="Always",
            task_id=f"clean_data",
            cmds=["bash", "-c"],
            arguments=["python clean_data.py"],
            on_finish_action='keep_pod',
            dag=dag,
        )

train_data = KubernetesPodOperator(
            image=f"hoangtuansu/trainer:{COMPONENT_IMAGE_VERSION}",
            env_vars=[
                    k8s.V1EnvVar(name='CLEAN_TRAIN_DATA_TABLE', value='TrainPM'),
                    k8s.V1EnvVar(name='CLEAN_TEST_DATA_TABLE', value='TestPM'),
                    k8s.V1EnvVar(name='CLEAN_ACTUAL_DATA_TABLE', value='ActualPM'),
                    k8s.V1EnvVar(name='MLFLOW_TRACKING_URI', value='http://10.180.113.115:32256'),
                    k8s.V1EnvVar(name='MLFLOW_TRACKING_USERNAME', value='user'),
                    k8s.V1EnvVar(name='MLFLOW_TRACKING_PASSWORD', value='sr9TvkIjaj')
                    ],
            name=f"train_data",
            task_id=f"train_data",
            image_pull_policy="Always",
            cmds=["bash", "-c"],
            arguments=["python model_train.py"],
            on_finish_action='keep_pod',
            dag=dag,
        )

ingest_data >> clean_data >> train_data