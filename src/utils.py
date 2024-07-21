import os
import logging
from constants import SERVICE_NAME


logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s {%(filename)s:%(lineno)d} - %(message)s','%Y-%m-%d:%H:%M:%S')

# create file handler which logs even debug messages
fh = logging.FileHandler(f'{os.getenv("FL_ROLE")}.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
