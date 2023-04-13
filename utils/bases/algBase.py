import os
import os.path as osp
from abc import *
import argparse 

from utils.loggers.logger import get_root_logger
from utils.bases.commBase import CommBase
from utils.helpers.parsing import set_cfgs, yaml2dict, set_augs

class AlgBase(CommBase, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.cfgs = argparse.Namespace() # external variables
        self._vars = argparse.Namespace() # internal variables
        self._flags = argparse.Namespace() # set internal flags
        self._augs = dict() # set aug parameters

        self.config_fp = None, 
        self.recipe_fp = None, 
        self.option_fp = None, 

        self._alg_status = "IDLE"
        self._alg_logger  = None 

    def _alg_set_status(self, status="", func_name="", class_name=""):
        self._alg_status = status
        self._alg_logger.debug(f"[{class_name}] - [{func_name}] - {self._alg_status}")

    def get_logger_handlers_info(self):
        return self._alg_logger.root.handlers

    def alg_log_info(self, info, func_name="", class_name=""):
        if self._alg_logger is not None:
            self._alg_logger.info(f"[{class_name}] - [{func_name}] - {self._alg_status} - {info}")

    def alg_log_debug(self, debug, func_name="", class_name=""):
        if self._alg_logger is not None:
            self._alg_logger.debug(f"[{class_name}] - [{func_name}] - {self._alg_status} - {debug}")

    def alg_log_error(self, err, func_name="", class_name=""):
        if self._alg_logger is not None:
            self._alg_logger.error(f"[{class_name}] - [{func_name}] - {self._alg_status} - {err}")

    def set_alg_logger(self, log_dir, log_level='info'):
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        self.comm_set_logger(osp.join(log_dir, "comm"))
        log_fname = osp.join(log_dir, "alg")
        while True:
            if osp.exists(log_fname):
                log_fname += "_resume"
            else:
                break

        self._alg_logger = get_root_logger(name=__class__.__name__, log_file=log_fname, log_level=log_level)

    @abstractmethod
    def alg_set_cfgs(self, config, recipe=None, option=None, mode=None, augmentations=None):
        '''
        read or get parameters or configurations
        '''
        self.config_fp = config 
        self.recipe_fp = recipe 
        self.option_fp = option 

        self.cfgs = set_cfgs(config=config, recipe=recipe, option=option, mode=mode)
        
        if augmentations: 
            self._augs = set_augs(augmentations=augmentations)
        # self._alg_set_status("GET cfgs.", self.alg_get_cfgs.__name__)

    def get_cfgs(self):
        return self.cfgs 

    def get_cfgs_config_dict(self):
        return yaml2dict(self.config_fp)
    
    def get_cfgs_recipe_dict(self):
        return yaml2dict(self.recipe_fp)

    def get_cfgs_option_dict(self):
        return yaml2dict(self.option_fp)

    @abstractmethod
    def alg_set_params(self):
        '''
        after getting or reading parameters, set parameters for speific task user wants
        '''
        # self._alg_set_status("SET params", self.alg_set_params.__name__)

    @abstractmethod
    def alg_set_datasets(self):
        self._alg_set_status("SET DATASETS", self.alg_set_datasets.__name__)

    @abstractmethod
    def alg_set_model(self):
        self._alg_set_status("SET MODEL", self.alg_set_model.__name__)

    @abstractmethod
    def alg_run_one_epoch(self):
        '''
        required to connect with backend
        '''
        self._alg_set_status("TRAIN ONE EPOCH", self.alg_run_one_epoch.__name__)

    @abstractmethod
    def alg_validate(self):
        '''
        required to connect with backend
        '''
        self._alg_set_status("VALIDATE", self.alg_validate.__name__)

    @abstractmethod
    def alg_end(self):
        self._alg_set_status("END", self.alg_end.__name__)
    
    @abstractmethod 
    def alg_reset(self):
        self._alg_set_status("RESET", self.alg_reset.__name__)

    @abstractmethod 
    def alg_stop(self):
        self._alg_set_status("STOP", self.alg_stop.__name__)