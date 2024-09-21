import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

DATA_PATH = os.getenv('DATA_PATH')