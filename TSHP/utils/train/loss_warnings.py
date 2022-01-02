import math

import numpy as np


class LossWarnings:
    def __init__(self, window_size: int, outlier_sensitivity: float, update_interval: int):
        """
        Will print warnings if any loss value in a batch is abnormally high.
        
        Can help identify bad audio files
        @param window_size: Number of items to use for calculating outlier ranges.
        @param outlier_sentivity: How easily a value should be considered an outlier. Recommend below 0.4 and above 0.01
        @param update_interval: How often to update the outlier ranges.
        """
        self.update_config(window_size, outlier_sensitivity, update_interval)
        self.loss_dict_list = {} # {loss_term: [old_value, older_value, ...], ...}
        self._iters = 0 # how many iterations have been ran so far
    
    def update_config(self, window_size: int, outlier_sensitivity: float, update_interval: int):
        self.window_size = int(window_size)
        self.outlier_sensitivity = float(outlier_sensitivity)
        self.update_interval = int(update_interval)
    
    def __call__(self, loss_dict_path: dict):# {path: {loss_term: loss_value, ...}, ...}
        if len(self.loss_dict_list.keys()) > 0 and len(next(iter(self.loss_dict_list.values()))) >= self.window_size:
            if self._iters%self.update_interval == 0 or not hasattr(self, 'max_acceptable_loss_dict'):# if X: recalculate max expected losses
                max_acceptable_loss_dict = {loss_term: np.quantile(loss_list, [0.98, 0.94]) for loss_term, loss_list in self.loss_dict_list.items()}
                max_acceptable_loss_dict = {loss_term: ((loss_list[0]-loss_list[1])/self.outlier_sensitivity)+loss_list[0] for loss_term, loss_list in max_acceptable_loss_dict.items()}
                self.max_acceptable_loss_dict = max_acceptable_loss_dict
            else:
                max_acceptable_loss_dict = self.max_acceptable_loss_dict
            
            # print warnings for paths/terms that are outside the expected range
            for path, loss_dict in loss_dict_path.items():
                outlier_losses = [loss_term for loss_term, loss_value in loss_dict.items() if loss_term in max_acceptable_loss_dict and (loss_value > max_acceptable_loss_dict[loss_term] or not math.isfinite(loss_value))]
                if len(outlier_losses):
                    print(f'"{path}" got outlier loss(es)\n'
                          f':Loss Term           :Loss Value          :Maximum Expected    :')
                    for loss_term in outlier_losses:
                        print(f':{loss_term:20}:{loss_dict[loss_term]:16.03f}:{max_acceptable_loss_dict[loss_term]:16.03f}')
        
        # add new loss values to list
        for path, loss_dict in loss_dict_path.items():
            for loss_term, loss_value in loss_dict.items():
                if not math.isfinite(loss_value): # if nan or inf, skip
                    continue
                if loss_term not in self.loss_dict_list:
                    self.loss_dict_list[loss_term] = []
                self.loss_dict_list[loss_term].append(loss_value)
                len_loss_list = len(self.loss_dict_list[loss_term])
                if len_loss_list > self.window_size:
                    self.loss_dict_list[loss_term] = self.loss_dict_list[loss_term][len_loss_list-self.window_size:]
        
        self._iters += 1
