import datetime
import time
import pandas as pd
import logging
import os
from utils import logger
from database import DATABASE


class IngestData(DATABASE):

    def __init__(self):
        super().__init__()
        self.connect()
        self.createdb(self.dbname)

    def createdb(self, dbname):
        if dbname not in self.client.get_list_database():
            logger.info(f"Create database: {dbname}")
            self.client.create_database(dbname)
            self.client.switch_database(dbname)

    def dropdb(self, dbname):
        if next((item for item in self.client.get_list_database() if item.get("name") == dbname), None) is not None:
            logger.info(f"DROP database: {dbname}")
            self.client.drop_database(dbname)
    
    def dropmeas(self, measname):
        logger.info(f"DROP MEASUREMENT: {measname}")
        self.client.query('DROP MEASUREMENT '+measname)

    def publish_db(self, df, meas):
        steps = df['measTimeStampRf'].unique()

        logger.info(f"Add data to measurement {meas}")
        
        for timestamp in steps:
            d = df[df['measTimeStampRf'] == timestamp]
            d.index = pd.date_range(start=datetime.datetime.now(), freq='1ms', periods=len(d))
            self.client.write_points(d, meas)
            time.sleep(0.5)
            

def main():
    # inintiate connection and create database UEDATA
    ingester = IngestData()
    df = pd.read_csv('ue.csv')
    table = os.getenv('RAW_DATA_TABLE')
    #data = ingester.query(f'select * from {table}')
    #ingester.dropmeas(table)
    df_sample = df.sample(frac=0.3)
    #ingester.publish_db(df_sample, table)
        

if __name__ == "__main__":
    main()