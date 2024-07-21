import time
import pandas as pd
from configparser import ConfigParser
from requests.exceptions import RequestException, ConnectionError
import os
from utils import logger


class Dataset(object):

    def __init__(self):
        self.data = []
        self.config()

    def augment_data(self, new_point):
        self.data.append(new_point)

    def config(self):
        cfg = ConfigParser()
        config_file_path = os.getenv('CONFIG_PATH')
        cfg.read(config_file_path)
        for section in cfg.sections():
            if section == 'features':
                self.thpt = cfg.get(section, "thpt")
                self.rsrp = cfg.get(section, "rsrp")
                self.rsrq = cfg.get(section, "rsrq")
                self.rssinr = cfg.get(section, "rssinr")
                self.prb = cfg.get(section, "prb_usage")
                self.ue = cfg.get(section, "ue")
                self.anomaly = cfg.get(section, "anomaly")
                self.a1_param = cfg.get(section, "a1_param")
