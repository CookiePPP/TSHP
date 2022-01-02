# General Imports
import os
import time
from copy import deepcopy
from datetime import datetime
from math import exp, log

import torch
import random
import traceback

import yaml

from TSHP.modules.train import LocalTrainModule
from TSHP.utils.misc_utils import deepupdate_dicts, create_nested_dict
from ax.service.ax_client import AxClient

import logging
_ = logging.getLogger('rich')
import TSHP.utils.warnings as w

# Config
MINIMIZE = True  # Whether we should be minimizing or maximizing the objective
METRIC = 'ax_target'

class AxModule:
    def __init__(self, args):
        self.args = args
        assert 'tune' in args.run_name.lower() or 'ax' in args.run_name.lower(), '"tune" or "ax" must be in run_name when performing tuning'
        with w.withLevel('warning'):
            tmp_trainmodule = LocalTrainModule(args.model, args.run_name, args.weight_name, args.active_dataset, args.rank, args.n_rank)
        self.run_path = tmp_trainmodule.run_path
        self.h = tmp_trainmodule.h
        
        # create AxClient object
        self.ax = AxClient(enforce_sequential_optimization=bool(args.n_rank == 1))# if n_rank is 1, use sequential optimization   
        
        self.sucessful_trials = 0
        self.failed_trials    = 0
    
    def sanity_checks(self):
        # check if max batch_size will OOM with the largest input batch
    
        # check if ...
        
        # check if MAE weights are positive
        
        # check if ACC weights are negative
        pass
    
    def load_experiment_params(self, config):
        """
        returns list of dicts with the parameter name, type, range, dtype and more.
        
        input: YAML style d, valid YAML example below;
              (the below would have to be loaded into a d before used with this func, I don't mean the literal YAML file string)
        \"""
        model_config:
            unet:
                n_stages:
                    ax_type: 'range'
                    ax_min: 4
                    ax_max: 12
                    ax_scale: 'linear'
        \"""
        
        e.g:
        return [
            {
                'name': 'learning_rate',
                'range': [1e-5, 1e-3],
                'dtype': 'int',
                'log_scale': True,
            }, ...
        ]
        @rtype: list[dict]
        """
        def check_ax(outlist, d, key):
            if 'ax_type' in d:
                if d['ax_type'] == 'range':
                    assert 'ax_min' in d, f'ax_min is missing from {key}'
                    assert 'ax_max' in d, f'ax_max is missing from {key}'
                    assert 'ax_scale' in d, f'ax_scale is missing from {key}'
                    assert 'ax_divisible_by' in d, f'ax_divisible_by is missing from {key}'
                    assert type(d['ax_min']) in [float, int], f'{key}.ax_min must be a number type'
                    assert type(d['ax_max']) in [float, int], f'{key}.ax_max must be a number type'
                    assert type(d['ax_min']) is type(d['ax_max']), f"{key}.ax_min and {key}.ax_max must be same type. Got {type(d['ax_min'])} and {type(d['ax_max'])}"
                    bounds = [d['ax_min'], d['ax_max']]
                    
                    assert not d['ax_divisible_by'] or d['ax_min'] % d['ax_divisible_by'] == 0, f'{key}.ax_min must be divisible by "ax_divisible_by"'
                    assert not d['ax_divisible_by'] or d['ax_max'] % d['ax_divisible_by'] == 0, f'{key}.ax_max must be divisible by "ax_divisible_by"'
                    if d['ax_divisible_by']:
                        print(f"before: {bounds}")
                        bounds = [x // d['ax_divisible_by'] for x in bounds]
                        print(f"after : {bounds}")
                    
                    value_type = str(type(d['ax_min'])).split("<class '")[-1].split("'>")[0]# "<class 'float'>" -> "float"
                    assert d['ax_scale'] in ['lin', 'linear', 'log', 'logarithmic', 'log+1', 'logarithmic+1'], f"ax_scale is invalid, got {d['ax_scale']}"
                    if d['ax_scale'] in ['log', 'logarithmic']:
                        assert d['ax_min'] > 0.0, f"{key}.ax_min must be greater than zero for log scale"
                        bounds = [log(x) for x in bounds]
                        value_type = 'float'
                    elif d['ax_scale'] in ['log+1', 'logarithmic+1']:
                        assert d['ax_min'] > -1.0, f"{key}.ax_min must be greater than '-1.0' for log+1 scale"
                        bounds = [log(x+1) for x in bounds]
                        value_type = 'float'
                    
                    return {
                        'name'      : key,
                        'type'      : 'range',
                        'bounds'    : bounds,
                        'value_type': value_type,# enforce float type if using log scaling
                        'real_value_type': str(type(d['ax_min'])).split("<class '")[-1].split("'>")[0], # actual value type
                        'ax_divisible_by': d['ax_divisible_by'],
                        'log_scale' : False,            # handle all lin/log/log+1 scaling on my side so I can add custom scaling rules
                        'ax_scale'  : d['ax_scale'], # handle all lin/log/log+1 scaling on my side so I can add custom scaling rules
                    }
                elif d['ax_type'] == 'choice':
                    assert 'ax_values' in d, f'ax_values is missing from {key}'
                    assert 'is_ordered' in d, f'is_ordered is missing from {key}'
                    assert type(d['ax_values']) in [list, tuple], f"{key}.ax_values must be a list"
                    dtype = type(d['ax_values'][0])
                    assert all(type(x) == dtype for x in d['ax_values']), f'{key}.ax_values has inconsistent data type'
                    return {
                        'name'      : key,
                        'type'      : 'choice',
                        'values'    : d['ax_values'],
                        'is_ordered': d['is_ordered'],
                    }
                else:
                    raise NotImplementedError
            elif any(x in d for x in ['ax_min', 'ax_max', 'ax_scale', 'ax_values']):
                raise Exception(f'ax_type is missing from {key}')
            else:
                for k, v in d.items():
                    if type(v) is type({}):
                        nkey = (f'{key}.{k}').lstrip('.')
                        param = check_ax(outlist, v, nkey)
                        if param is not None:
                            outlist.append(param)
        
        # unpack nested d structure into list of parameters
        parameters = []
        check_ax(parameters, config, '')# inplace modification of param_list
        
        # filter out items for this module and items for create_experiment
        self.exp_parameters = parameters # save for my wrapping stuff
        parameters = [{k: v for k, v in d.items() if k not in ['ax_scale','ax_divisible_by','real_value_type']} for d in parameters] # send to ax_client
        return parameters # returns list of dicts, each d has the name of a param along with it's range, dtype and other stuff
    
    def set_experiment_params(self, parameters):
        # Configure the Hyperparameters for Tuning
        # Set ranges, dtypes and sampling skews
        self.ax.create_experiment(
            name="ax_experiment",
            parameters=parameters,
            minimize=MINIMIZE,
            objective_name=METRIC,
        )
    
    def save_state(self, path):
        pass
    
    def load_state(self, path):
        pass
    
    def undo_scaling(self, axparams):
        # perform any value scaling that ax cannot do
        axparams = {k: v for k, v in axparams.items()} # copy dictionary before performing inplace modification
        expparams = {x['name']: x for x in self.exp_parameters}
        for k, d in expparams.items():
            if k in axparams and d['type'] == 'range':
                if d['ax_scale'] in ['log', 'logarithmic']:
                    # reverse the log() done on the experiment_parameters
                    axparams[k] = exp(axparams[k])
                elif d['ax_scale'] in ['log+1', 'logarithmic+1']:
                    # reverse the log() done on the experiment_parameters
                    axparams[k] = exp(axparams[k])-1
                
                if d['ax_divisible_by']:
                    # reverse the divide done on the experiment_parameters
                    axparams[k] = round(axparams[k])*d['ax_divisible_by']
                
                if d['real_value_type'] == 'int':
                    axparams[k] = round(axparams[k])
        return axparams
    
    def export_axparams(self, axparams):
        # perform any value scaling that ax cannot do
        axparams = self.undo_scaling(axparams)
        
        # convert from axparams to nested model_config dict
        ax_model_config = create_nested_dict(axparams)# convert dict to nested dict
                                    # e.g:  {'a.b.c': 'val'} -> {'a': {'b': {'c': 'val'}}}
        return ax_model_config
    
    def convert_axparams_to_config(self, axparams):
        ax_model_config = self.export_axparams(axparams)
        
        #config = {k: v for k, v in self.h.items()} # copy config
        #deepupdate_dicts(config, ax_model_config)
        return ax_model_config
    
    def multiprocess_evaluate(self, model_config):
        """MultiGPU version of evaluate for use with run_trial()"""
        return results
    
    def evaluate(self, config, axparams):
        #class ArgsClass:
        #    pass
        #
        #args = ArgsClass()
        #args.__setattr__('output_directory', self.args.output_directory)
        
        # initialize TrainModule
        override_config = deepcopy(self.h)
        deepupdate_dicts(override_config, config, warn_if_key_not_exists=True)
        override_config['metricmodule_config']['tensorboardlogger_config']['tuning'] = True
        with w.withLevel('warning'):
            trainmodule = LocalTrainModule(self.args.model, self.args.run_name, self.args.weight_name, self.args.active_dataset, self.args.rank, self.args.n_rank, override_config=override_config)
        
        # train model and evaluate
        # results = {f'{train/cross_val/test}__{lossterm}': loss_value, ...}
        trainmodule.get_max_learning_rate(1e-5, 1e-3, 1_000_000, file_window_size=8192)
        trainmodule.train_till(**self.h['tuning_config']['stop_config'])
        results = trainmodule.metricmodule.bestepochavg_loss_dict
        
        # calculate ax_target value
        loss_weights = self.h['tuning_config']['loss_weights']
        if not all(k in results for k in loss_weights.keys()):
            w.print1("Keys below are missing from results.")
            w.print1(" -"+"\n -".join([k for k in loss_weights.keys() if k not in results]))
        results['ax_target'] = sum(v*loss_weights.get(k, 0.0) for k, v in results.items() if k in loss_weights)
        
        return results
    
    def run_trial(self):
        parameters, trial_index = self.ax.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        config = self.convert_axparams_to_config(parameters)
        w.print1("Running new trial with parameters below:")
        pretty_print(self.export_axparams(parameters))
        w.print1("\n\n")
        
        try:
            results = self.evaluate(config, self.undo_scaling(parameters)) # results should be a dict of {lossname: loss value, ...}    
        except Exception as ex:
            traceback.print_exc()
            time.sleep(5.0)
            results = None
        if results is None:
            self.ax.log_trial_failure(trial_index=trial_index)
            self.failed_trials += 1
        else:
            self.ax.complete_trial(trial_index=trial_index, raw_data=results)
            self.sucessful_trials += 1
                    

def pretty_print(d, indent=0):
   for key, value in d.items():
      w.print1('\t' * indent + str(key) + ':')
      if isinstance(value, dict):
         pretty_print(value, indent+1)
      else:
         w.print1('\t' * (indent+1) + str(value))
