
# !!!!! We must set this environment variable before importing asyncio !!!!!
import os
from dataset import Dataset
import tempfile
from typing import Dict
from constants import COUNTER_NAMES, SERVICE_NAME

os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix=str(os.getpid()))

# we must set_start_method before import asyncio to get data processing to
# work on a mac. This code will _not_ work on Windows.
import multiprocessing  # noqa: E402

multiprocessing.set_start_method("fork")

import asyncio  # noqa: E402
import json
import os
import time
import pandas as pd
import schedule
from ad_model import modelling, CAUSE
from ad_train import ModelTraining
from dataset import Dataset
from utils import logger
from pmconsumer import PmHistoryConsumer

db = None
cp = None
threshold = None


def load_model():
    global md
    global cp
    global threshold
    md = modelling()
    cp = CAUSE()
    threshold = 70
    logger.info("throughput threshold parameter is set as {}% (default)".format(threshold))


def train_model(ds):
    if not os.path.isfile('/opt/oran/src/model'):
        mt = ModelTraining(ds)
        mt.train()


def predict(self):
    db.read_data()
    val = None
    if db.data is not None:
        if set(md.num).issubset(db.data.columns):
            db.data = db.data.dropna(axis=0)
            if len(db.data) > 0:
                val = predict_anomaly(self, db.data)
        else:
            logger.warning("Parameters does not match with of training data")
    else:
        logger.warning("No data in last 1 second")
        time.sleep(1)
    if (val is not None) and (len(val) > 2):
        msg_to_ts(self, val)


def predict_anomaly(self, df):
    df['Anomaly'] = md.predict(df)
    df.loc[:, 'Degradation'] = ''
    val = None
    if 1 in df.Anomaly.unique():
        df.loc[:, ['Anomaly', 'Degradation']] = cp.cause(df, db, threshold)
        df_a = df.loc[df['Anomaly'] == 1].copy()
        if len(df_a) > 0:
            df_a['time'] = df_a.index
            cols = [db.ue, 'time', 'Degradation']
            # rmr send 30003(TS_ANOMALY_UPDATE), should trigger registered callback
            result = json.loads(df_a.loc[:, cols].to_json(orient='records'))
            val = json.dumps(result).encode()
    df.loc[:, 'RRU.PrbUsedDl'] = df['RRU.PrbUsedDl'].astype('float')
    df['Viavi.UE.anomalies'] = df['Viavi.UE.anomalies'].astype('int64')
    df['du-id'] = df['du-id'].astype('int64')

    df.index = pd.date_range(start=df.index[0], periods=len(df), freq='1ms')
    db.write_anomaly(df)
    return val


def msg_to_ts(self, val):
    logger.debug("Detecting Anomalous UE")


async def main(self):
    pmhistory = PmHistoryConsumer(logger)
    await pmhistory.new_rapp_session()

    dataset = Dataset()
    train_model(dataset)
    load_model()
    schedule.every(0.5).seconds.do(predict, self)
    while True:
        schedule.run_pending()


if __name__ == "__main__":
    asyncio.run(main())