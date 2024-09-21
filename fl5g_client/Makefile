SHELL=/bin/bash

all:
	export COMPONENT_IMAGE_VERSION=0.3

	export DAG_IMAGE_VERSION=0.4

	cd /home/system/workspace/vmware/afl_oran_traffic_prediction/data_pipeline/

	docker-compose build ingester cleaner trainer evaluator

	docker push hoangtuansu/ingester:$COMPONENT_IMAGE_VERSION

	docker push hoangtuansu/cleaner:$COMPONENT_IMAGE_VERSION

	docker push hoangtuansu/trainer:$COMPONENT_IMAGE_VERSION


	docker-compose build dagger && docker push hoangtuansu/ad_dag:$DAG_IMAGE_VERSION && helm upgrade airflow apache-airflow/airflow --namespace taisp-ml-infra --set images.airflow.repository=hoangtuansu/ad_dag --set images.airflow.tag=$DAG_IMAGE_VERSION --set images.airflow.pullPolicy=Always

