from datetime import datetime
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from database import DATABASE
from processing import PREPROCESS
from utils import logger, DATA_PATH


def processing_data(train_data, test_data):
    logger.debug("Processing training data")
    ps1 = PREPROCESS(train_data)
    ps1.process()

    actual_data = (test_data['Viavi.UE.anomalies'] > 0).astype(int)

    num = joblib.load(f'{DATA_PATH}/num_params')
    ps2 = PREPROCESS(test_data[num])
    ps2.transform()
    return ps1.data, ps2.data, actual_data


def main():
    try:
        db = DATABASE()
        success = False
        while not success:
            success = db.connect()

        raw_table = os.getenv('RAW_DATA_TABLE')
        clean_train_table = os.getenv('CLEAN_TRAIN_DATA_TABLE')
        clean_test_table = os.getenv('CLEAN_TEST_DATA_TABLE')
        clean_actual_table = os.getenv('CLEAN_ACTUAL_DATA_TABLE')

        logger.info(f'Raw data table: {raw_table}, clean train table: {clean_train_table}, clean test table: {clean_test_table}, actual table: {clean_actual_table}')

        raw_data = db.query(f'select * from {raw_table}')[raw_table]
        db.query('DROP MEASUREMENT ' + clean_train_table)
        db.query('DROP MEASUREMENT ' + clean_test_table)
        db.query('DROP MEASUREMENT ' + clean_actual_table)

        train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
        processed_train_data, processed_test_data, actual_data = processing_data(train_data, test_data)

        processed_train_data.index = pd.date_range(start=datetime.now(), periods=len(processed_train_data), freq='1ms')
        db.client.write_points(processed_train_data, clean_train_table)

        processed_test_data.index = pd.date_range(start=datetime.now(), periods=len(processed_test_data), freq='1ms')
        db.client.write_points(processed_test_data, clean_test_table)

        actual_data.index = pd.date_range(start=datetime.now(), periods=len(actual_data), freq='1ms')
        db.client.write_points(actual_data.to_frame(), clean_actual_table)

        return
    except Exception as e:
        logger.error(e)
        raise e
    

if __name__ == "__main__":
    main()