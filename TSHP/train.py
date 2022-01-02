import argparse
import logging
import math
import os
import signal

import torch.autograd

from CookieSpeech.modules.train import GlobalTrainModule

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
    parser.add_argument('--detect_anomaly', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='detects NaN/Infs in autograd backward pass and gives additional debug info.')
    parser.add_argument('--dump_alignments', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reset_static_keys', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reset_best_cross_val_stats', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reset_lr_schedule_state', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_secpr', type=float, default=1_000_000)
    parser.add_argument('--patience_secpr', type=float, default=1_000_000)
    parser.add_argument('--secpr_schedule_divider', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=float('nan'))
    parser.add_argument('-l', '--logging_level', type=str, default='info',
                        help="'options: ['debug', 'info', 'warning', 'error', 'critical']")
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('-ad', '--active_dataset', type=str, default=None,
                        required=False, help='active_dataset override')
    args = parser.parse_args()
    print(args)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    if args.active_dataset == '':
        args.active_dataset = None
    if args.weight_name == '':
        args.weight_name = None
    if args.weight_name == '':
        args.weight_name = None
    if args.patience_secpr == '':
        args.patience_secpr = None
    if args.patience_secpr == '':
        args.patience_secpr = None
    
    w.setLevel(args.logging_level.upper())
    
    try:
        trainmodule = GlobalTrainModule(args.model, args.run_name, args.weight_name, args.active_dataset, args.rank, args.n_gpus, args=args)
        
        if not math.isnan(args.lr):
            trainmodule.modelmodule.model.max_learning_rate.fill_(args.lr)
        
        if args.reset_static_keys:
            trainmodule.metricmodule.last_plot_times = {}
            trainmodule.metricmodule.static_keys = {}
        
        if args.reset_best_cross_val_stats:
            trainmodule.metricmodule.last_plot_times = {}
            trainmodule.metricmodule.static_keys = {}
            trainmodule.metricmodule.epochavg_dict = {}
            trainmodule.metricmodule.expavg_loss_dict = {}
            trainmodule.metricmodule.bestexpavg_loss_dict = {}
            trainmodule.metricmodule.bestepochavg_loss_dict = {}
            trainmodule.modelmodule.model.best_cross_val_secpr.fill_(trainmodule.modelmodule.model.secpr + 10_000)
            trainmodule.modelmodule.model.best_cross_val.fill_(float('inf'))
        
        if args.reset_lr_schedule_state:
            trainmodule.modelmodule.model.lr_multiplier.fill_(1.0)
        
        if args.dump_alignments:
            trainmodule.dump_features(
                pr_key='alignments',
                file_extension='_align.pt',
                arpabet=False,
            )
            trainmodule.dump_features(
                pr_key='alignments',
                file_extension='_plign.pt',
                arpabet=True,
            )
        else:
            if trainmodule.modelmodule.model.max_learning_rate == 0.0:
                trainmodule.get_max_learning_rate(1e-7, 1e-1, args.warmup_secpr//args.secpr_schedule_divider, file_window_size=8192//args.secpr_schedule_divider)
            trainmodule.train_till(
                secpr=300_000_000//args.secpr_schedule_divider,
                patience_secpr=args.patience_secpr//args.secpr_schedule_divider if args.patience_secpr is not None else None
            )
    
    except KeyboardInterrupt:
        if args.n_gpus > 1:
            for p in trainmodule.workers:
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)