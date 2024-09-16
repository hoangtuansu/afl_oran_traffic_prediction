from datetime import datetime
import time
import mlflow
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from database import DATABASE
from utils import logger


mlflow.set_tracking_uri("http://10.180.113.115:32256")
mlflow.set_experiment('ORAN PM Training')

actual_data = None
test_data = None

def validate(model, x_test):
    pred = model.predict(test_data.values)
    if -1 in pred:
        pred = [1 if p == -1 else 0 for p in pred]
    F1 = f1_score(actual_data, pred, average='macro')
    logger.debug("classfication report : {} ".format(classification_report(actual_data, pred)))
    logger.debug("F1 score:{}".format(F1))
    return F1

def isoforest(train_data, test_data, random_state=4):
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"ad_training_{run_name}"):
        mlflow.sklearn.autolog()
        logger.info((mlflow.get_tracking_uri(), mlflow.get_artifact_uri()))
        parameter = {'contamination': [of for of in np.arange(0.01, 0.5, 0.02)],
                        'n_estimators': [100*(i+1) for i in range(1, 10)],
                        'max_samples': [0.005, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4]}
        
        mlflow.log_param("random_state", 4)
        mlflow.log_param("contamination", parameter['contamination'])
        mlflow.log_param("n_estimators", parameter['n_estimators'])
        mlflow.log_param("max_samples", parameter['max_samples'])

        cv = [(slice(None), slice(None))]
        iso = IsolationForest(random_state=random_state, bootstrap=True, warm_start=False)
        model = RandomizedSearchCV(iso, parameter, scoring=validate, cv=cv, n_iter=50)
        md = model.fit(train_data.values)
        f1 = validate(md.best_estimator_, test_data)
        mlflow.log_metric("F1 score", f1)
        mlflow.log_metric("Best score", md.best_score_)


def main():
    try:
        global actual_data
        global test_data

        db = DATABASE()
        success = False
        while not success:
            success = db.connect()

        clean_train_table = os.getenv('CLEAN_TRAIN_DATA_TABLE')
        clean_test_table = os.getenv('CLEAN_TEST_DATA_TABLE')
        clean_actual_table = os.getenv('CLEAN_ACTUAL_DATA_TABLE')

        train_data = db.query(f'select * from {clean_train_table}')[clean_train_table]
        test_data = db.query(f'select * from {clean_test_table}')[clean_test_table]
        actual_data = db.query(f'select * from {clean_actual_table}')[clean_actual_table]

        logger.debug("Training Starts")
        isoforest(train_data, test_data)
        time.sleep(20)
        return
    except Exception as e:
        logger.error(e)
        time.sleep(20)
        raise e
    

if __name__ == "__main__":
    main()