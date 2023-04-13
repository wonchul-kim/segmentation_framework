import os
import os.path as osp
from abc import *
from this import d
import threading 

from aivutils.loggers.logger import get_root_logger

class CommBase(object):
    '''
        This class is for communication with backend(django server) to control threading event 
        Status: 
            * IDLE: keep going algorithm
            * SET: stop the algorithm
    '''
    def __init__(self):
        self._comm_threading_event = threading.Event() 
        self._comm_logger = None 
        self._comm_status = None

    def _comm_set_status(self, status="", func_name=""):
        self._comm_status = status 
        self._comm_logger.info(f"[{func_name}] - {self._comm_status}")

    def comm_log_info(self, info, func_name=""):
        if self._comm_logger is not None:
            self._comm_logger.info(f"[{func_name}] - {self._comm_status} - {info}")

    def comm_log_error(self, err, func_name=""):
        if self._comm_logger is not None:
            self._comm_logger.error(f"[{func_name}] - {self._comm_status} - {err}")

    def comm_set_logger(self, log_file):
        self._comm_logger = get_root_logger(name=__class__.__name__, log_file=log_file)
        self._comm_set_status("IDLE", self.comm_set_logger.__name__)

    def comm_reset(self):
        self._comm_threading_event.clear() 
        self._comm_set_status("IDLE", self.comm_reset.__name__)
    
    def comm_stop(self):
        self._comm_threading_event.set() 
        self._comm_set_status("SET", self.comm_stop.__name__)
        self.comm_reset()

    def comm_keep_going(self):
        if self._comm_status == 'IDLE':
            return True 
        elif self._comm_status == "SET": 
            return False
        else:
            self.comm_log_error("There is no such error", self.comm_keep_going.__name__)
            raise NotImplementedError(f"There is no such error: {self.comm_keep_going.__name__}")
