import logging
import os
from utils import get_local_time

def init_logger(config):
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(message)s"
    fileformatter = logging.Formatter(filefmt)

    sfmt = "%(message)s"
    sformatter = logging.Formatter(sfmt)
    
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
        
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[sh, fh]
    )

def getLogger():
    return logging.getLogger()