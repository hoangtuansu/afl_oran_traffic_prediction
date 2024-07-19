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
import time
import pandas as pd
from influxdb import DataFrameClient
from configparser import ConfigParser
from mdclogpy import Logger
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError
from requests.exceptions import RequestException, ConnectionError
import os

logger = Logger(name=__name__)


class DATABASE(object):
    r""" DATABASE takes an input as database name. It creates a client connection
      to influxDB and It reads/ writes UE data for a given dabtabase and a measurement.


    Parameters
    ----------
    host: str (default='r4-influxdb.ricplt.svc.cluster.local')
        hostname to connect to InfluxDB
    port: int (default='8086')
        port to connect to InfluxDB
    username: str (default='root')
        user to connect
    password: str (default='root')
        password of the use

    Attributes
    ----------
    client: influxDB client
        DataFrameClient api to connect influxDB
    data: DataFrame
        fetched data from database
    """

    def __init__(self):
        self.data = None
        self.client = None
        self.config()

    def connect(self):
        if self.client is not None:
            self.client.close()

        try:
            self.client = DataFrameClient(self.host, port=self.port, username=self.user, password=self.password, database=self.dbname)
            version = self.client.request('ping', expected_response_code=204).headers['X-Influxdb-Version']
            logger.info("Conected to Influx Database, InfluxDB version : {}".format(version))
            return True

        except (RequestException, InfluxDBClientError, InfluxDBServerError, ConnectionError):
            logger.error("Failed to establish a new connection with InflulxDB, Please check your url/hostname")
            time.sleep(120)

    def read_data(self, train=False, valid=False):
        """Read data method for a given measurement and limit

        Parameters
        ----------
        meas: str (default='ueMeasReport')
        limit:int (defualt=False)
        """
        self.data = None

        if not train and not valid:
            query = f'select * from {self.meas} where time>now()-1600ms'
        elif train:
            query = f'select * from {self.meas} where time<now()-5m and time>now()-75m'
        elif valid:
            query = f'select * from {self.meas} where time>now()-5m'

        result = self.query(query)
        if result and len(result[self.meas]) != 0:
            self.data = result[self.meas]

    def write_anomaly(self, df, meas='AD'):
        """Write data method for a given measurement

        Parameters
        ----------
        meas: str (default='AD')
        """
        try:
            self.client.write_points(df, meas)
        except (RequestException, InfluxDBClientError, InfluxDBServerError) as e:
            logger.error('Failed to send metrics to influxdb')
            print(e)

    def query(self, query):
        try:
            result = self.client.query(query)
        except (RequestException, InfluxDBClientError, InfluxDBServerError, ConnectionError) as e:
            logger.error('Failed to connect to influxdb: {}'.format(e))
            result = False
        return result

    def config(self):
        cfg = ConfigParser()
        config_file_path = os.getenv('XAPP_CONFIG_PATH')
        cfg.read(config_file_path)
        for section in cfg.sections():
            if section == 'influxdb':
                self.host = cfg.get(section, "host")
                self.port = cfg.get(section, "port")
                self.user = cfg.get(section, "user")
                self.password = os.getenv('INFLUXDB_PASSWORD')
                self.dbname = cfg.get(section, "database")
                self.meas = cfg.get(section, "table")

            if section == 'features':
                self.thpt = cfg.get(section, "thpt")
                self.rsrp = cfg.get(section, "rsrp")
                self.rsrq = cfg.get(section, "rsrq")
                self.rssinr = cfg.get(section, "rssinr")
                self.prb = cfg.get(section, "prb_usage")
                self.ue = cfg.get(section, "ue")
                self.anomaly = cfg.get(section, "anomaly")
                self.a1_param = cfg.get(section, "a1_param")
