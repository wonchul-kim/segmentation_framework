import os
import os.path as osp
import pandas as pd 
import matplotlib.pyplot as plt 

class Monitor(object):
    """
    Since using pandas as default, 
    setting columns should be list and adding data should be dict.
    """
    def __init__(self, output_dir, fn, ext='csv'):
        self._output_dir = output_dir 
        self._fn = fn
        self._ext = ext
        self._info = {}

    def log(self, data):
        if isinstance(data, dict):
            for key, val in data.items():
                if key not in self._info.keys():
                    self._info[key] = [val]
                else:
                    self._info[key].append(val)
        else:
            raise TypeError(f"Logging data should be dict, not {type(data)}")
    
    def save(self, figs=False):
        if self._ext == 'csv':
            _df = pd.DataFrame.from_dict(self._info)
            if not osp.exists(self._output_dir):
                os.mkdir(self._output_dir)
            _df.to_csv(osp.join(self._output_dir, self._fn + '.csv'), index=False)

            if figs:
                for key, val in self._info.items():
                    fig = plt.figure(figsize=(20, 10))
                    plt.plot(range(len(self._info[key])), self._info[key])
                    plt.ylabel(key)
                    plt.suptitle(key, fontsize=20)
                    plt.savefig(osp.join(self._output_dir, '{}_{}.png'.format(self._fn, key)))
                    plt.close()
        else:
            raise TypeError(f"Cannot save that format: {self._ext}")        

if __name__ == '__main__':
    monitor = Monitor('./', 'test')
    # monitor.set(["epoch", "loss", "acc"])
    monitor.log({"acc": 0, "loss": 0.1})
    monitor.log({"acc": 4, "loss": 0.001})
    monitor.log({"acc": 1, "loss": 0.00001})
    # monitor.log({'min': 11})
    monitor.save()

    # print(monitor._info)
    