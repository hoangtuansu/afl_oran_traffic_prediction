import time
import logging
from influxdb import DataFrameClient
from configparser import ConfigParser
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError
from requests.exceptions import RequestException, ConnectionError
import os
from utils import logger



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
        cfg = ConfigParser()
        self.host = 'influxdb.ricinfra'
        self.port = '8086'
        self.user = 'admin'
        self.password = os.getenv('INFLUXDB_PASSWORD')
        self.dbname = 'ORAN'

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

    def read_data(self, meas, train=False, valid=False):
        """Read data method for a given measurement and limit

        Parameters
        ----------
        meas: str (default='ueMeasReport')
        limit:int (defualt=False)
        """ 
        self.data = None

        if not train and not valid:
            query = f'select * from {meas} where time>now()-1600ms'
        elif train:
            query = f'select * from {meas} where time<now()-5m and time>now()-75m'
        elif valid:
            query = f'select * from {meas} where time>now()-5m'

        result = self.query(query)
        if result and len(result[meas]) != 0:
            self.data = result[meas]

    def write_anomaly(self, df, meas):
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