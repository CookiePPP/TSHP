import argparse
import os

from CookieSpeech.utils.ax.hparam_tune import AxModule, pretty_print

import logging
log = logging.getLogger('rich')
import CookieSpeech.utils.warnings as w

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='the model to use. E.g: "DiffWave" or "Tacotron2"')
    parser.add_argument('-o', '--run_name', type=str, default='outdir',
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--weight_name', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='detects NaN/Infs in autograd backward pass and gives additional debug info.')
    parser.add_argument('-l', '--logging_level', type=str, default='info',
                        help="'options: ['debug', 'info', 'warning', 'error', 'critical']")
    parser.add_argument('--n_rank', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('-ad', '--active_dataset', type=str, default=None,
                        required=False, help='active_dataset override')
    args = parser.parse_args()
    
    w.setLevel(args.logging_level.upper())
    
    # get check this GPU is not busy
    pass
    
    # start tuning Module
    axmodule = AxModule(args)
    
    parameters = axmodule.load_experiment_params(axmodule.h['trainmodule_config'])
    axmodule.set_experiment_params(parameters)
    for i in range(axmodule.h['tuning_config']['n_trials']):
        axmodule.run_trial()
        
        ax_results_path = os.path.join(axmodule.run_path, 'ax_results.json')
        os.makedirs(os.path.split(ax_results_path)[0], exist_ok=True)
        axmodule.ax.save_to_json_file(ax_results_path)# save results
        
        print('\n')
        print(f'{axmodule.sucessful_trials} Sucessful Trials')
        print(f'{axmodule.failed_trials} Failed Trials')
        
        try:
            best_parameters, values = axmodule.ax.get_best_parameters()
            best_parameters = axmodule.export_axparams(best_parameters)
            print("\n[Best Parameters]")
            _=pretty_print(best_parameters)
            print("\n[Metrics]")
            _=pretty_print(values[0])
        except:
            pass
        print("\n")