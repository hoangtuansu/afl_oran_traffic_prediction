import joblib
import time
import numpy as np
import pandas as pd
from dataset import Dataset
from processing import PREPROCESS
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
from utils import logger


class ModelTraining(object):
    r""" The modelling class takes input as dataframe or array and train Isolation Forest model

    Paramteres
    .........
    data: DataFrame or array
        input dataset
    cols: list
        list of parameters in input dataset

    Attributes
    ----------
    actual:array
        actual label for test data
    X: DataFrame or array
        transformed values of input data
    """
    def __init__(self, ds):
        self.dataset : Dataset = ds
        self.train_data = None
        self.test_data = None
        self.read_train()
        self.read_test()

    def read_train(self):
        while self.dataset.data is None or\
              len(self.dataset.data) < 1000:
            logger.warning("Not sufficient data for Training. Sleep 120 seconds before checking data again.")
            time.sleep(120)
        self.train_data = pd.DataFrame(self.dataset.data)
        logger.debug(f"Training on {len(self.train_data)} samples")

    def read_test(self):
        """ Read test dataset for model validation"""
        while self.dataset.data is None or len(self.dataset.data) < 300:
            logger.warning("Check if InfluxDB instance is up? or Not sufficient data for Validation in last 10 minutes")
            time.sleep(60)
        self.test_data = pd.DataFrame(self.dataset.data)
        logger.debug(f"Validation on {len(self.test_data)} Samples")

    def isoforest(self, outliers_fraction=0.05, random_state=4):
        """ Train isolation forest

        Parameters
        ----------
        outliers_fraction: float between 0.01 to 0.5 (default=0.05)
            percentage of anomalous available in input data
        push_model: boolean (default=False)
            return f_1 score if True else push model into repo
        random_state: int (default=42)
        """
        parameter = {'contamination': [of for of in np.arange(0.01, 0.5, 0.02)],
                     'n_estimators': [100*(i+1) for i in range(1, 10)],
                     'max_samples': [0.005, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4]}
        cv = [(slice(None), slice(None))]
        iso = IsolationForest(random_state=random_state, bootstrap=True, warm_start=False)
        model = RandomizedSearchCV(iso, parameter, scoring=self.validate, cv=cv, n_iter=50)
        md = model.fit(self.train_data.values)
        f1 = self.validate(md.best_estimator_, self.test_data, True)
        return f1, md.best_estimator_

    def validate(self, model, test_data, report=False):
        pred = model.predict(self.test_data.values)
        if -1 in pred:
            pred = [1 if p == -1 else 0 for p in pred]
        F1 = f1_score(self.actual, pred, average='macro')
        if report:
            logger.debug("classfication report : {} ".format(classification_report(self.actual, pred)))
            logger.debug("F1 score:{}".format(F1))
        return F1

    def train(self):
        """
        Main function to perform training on input data
        """
        logger.debug("Training Starts")
        ps = PREPROCESS(self.train_data)
        ps.process()
        self.train_data = ps.data

        self.actual = (self.test_data[self.db.anomaly] > 0).astype(int)
        num = joblib.load('/opt/oran/src/num_params')
        ps = PREPROCESS(self.test_data[num])
        ps.transform()
        self.test_data = ps.data

        scores = []
        models = []

        logger.info("Training Isolation Forest")
        f1, model = self.isoforest()
        scores.append(f1)
        models.append(model)

        opt = scores.index(max(scores))
        joblib.dump(models[opt], '/opt/oran/src/model')
        logger.info("Optimum f-score : {}".format(scores[opt]))
        logger.info("Training Ends : ")
