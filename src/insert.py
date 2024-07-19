# ==================================================================================
#  Copyright (c) 2020 HCL Technologies Limited.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==================================================================================

"""
This Module is temporary for pushing data into influxdb before dpeloyment of AD xApp. It will depreciated in future, when data will be coming through KPIMON
"""

import datetime
import time
import sys
import pandas as pd
from database import DATABASE
from mdclogpy import Logger

logger = Logger(name=__name__)

class INSERTDATA(DATABASE):

    def __init__(self):
        super().__init__()
        self.config()
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

    def assign_timestamp(self, df):
        steps = df['measTimeStampRf'].unique()

        logger.info(f"Add data to measurement {self.meas}")
        
        for timestamp in steps:
            d = df[df['measTimeStampRf'] == timestamp]
            d.index = pd.date_range(start=datetime.datetime.now(), freq='1ms', periods=len(d))
            self.client.write_points(d, self.meas)
            time.sleep(0.7)


def main():
    # inintiate connection and create database UEDATA
    db = INSERTDATA()
    df = pd.read_csv('ue.csv')   
    while True:
        db.assign_timestamp(df)
        db.dropmeas(db.meas)


if __name__ == "__main__":
    main()