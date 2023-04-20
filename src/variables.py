

def set_variables(self):
    if self._var_ml_framework == 'pytorch':
        pass
    elif self._var_ml_framework == 'tensorflow':
        from frameworks.tensorflow.src.variables import set_variables as set_tensorflow_variables 
        
        self._vars.device_ids, self._var_strategy = set_tensorflow_variables(self._vars.device, self._vars.device_ids, True)
                