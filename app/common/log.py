# -*- coding: utf-8 -*-
import os
import sys
import logging

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = os.path.join(curdir, os.path.pardir, os.path.pardir, 'logs', 'root.log')
print(f'Saving logs into {log_file}')
LOG_LEVEL = "INFO"
fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
fh.setLevel(LOG_LEVEL)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(LOG_LEVEL)


def set_log_level(level="DEBUG"):
    fh.setLevel(level)
    ch.setLevel(level)


def getLogger(logger_name, level="DEBUG"):
    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(level)  # logging.DEBUG
    return logger


logger = getLogger('root')

if __name__ == "__main__":
    logger = getLogger('foo')
    logger.info('bar')
