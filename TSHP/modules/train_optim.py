import math
import traceback
from typing import List, Optional, Union, Dict, Tuple, Any

import torch
from TSHP.utils.modules.core import nnModule
from TSHP.utils.saving.utils import safe_write

import logging
log = logging.getLogger('rich')
import TSHP.utils.warnings as w

# these are needed for fp16 training, not inference
try:
    from apex import amp
    from apex import optimizers as apexopt
    def apexopt_FusedLAMB(*args, **kwargs):
        return apexopt.FusedLAMB(*args, **kwargs, max_grad_norm=9999.9)
except ImportError:
    apexopt = None
    class amp: # make mock amp if AMP is not installed
        def master_params(*args, **kwargs):
            raise NotImplementedError
        
        def state_dict(*args, **kwargs):
            raise NotImplementedError
        
        def scale_loss(*args, **kwargs):
            raise NotImplementedError
        
        def initialize(*args, **kwargs):
            raise NotImplementedError

# OptimizerModule
def get_optimizer(source='pytorch', opt_name='Adam'):
    source, opt_name = source.lower(), opt_name.lower()  # set inputs to lowercase
    if source == 'pytorch':
        if opt_name == 'adam':
            return torch.optim.Adam
        elif opt_name == 'lamb':
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif source == 'apex':
        if opt_name == 'adam':
            return apexopt.FusedAdam
        elif opt_name == 'lamb':
            return apexopt_FusedLAMB
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # test get_optimizer()
    opt = get_optimizer('pytorch', 'adam')
    assert opt is not None
    assert hasattr(opt, 'step')
    
    # should trigger NotImplementedError
    try:
        opt = get_optimizer('pytorch', 'lamb')
        assert opt is not None
        assert hasattr(opt, 'step')
        raise AssertionError
    except AssertionError:
        raise
    except NotImplementedError:
        pass
    
    # (only test if Apex Optimizers are installed)
    if apexopt is not None:
        opt = get_optimizer('apex', 'adam')
        assert opt is not None
        assert hasattr(opt, 'step')
        
        opt = get_optimizer('apex', 'lamb')
        assert opt is not None
        assert hasattr(opt, 'step')
    
    w.print1('get_optimizer() Test Completed!')

def clip_grad_norm_(model, optimizer, grad_clip_thresh, fp16_run=False):
    is_overflow = False
    if grad_clip_thresh is not None:  # apply gradient clipping to params
        if fp16_run:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), grad_clip_thresh)
            is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            if not math.isfinite(grad_norm):
                w.print2(f'Rank NaN Gradnorm(s) from;\n'+
                      '\n'.join([f'- {key} - {torch.norm(p.grad.detach(), 2.0).mean()} - {torch.norm(p.grad.detach(), 2.0).sum()} - {p.grad.detach().mean()} - {p.grad.detach().sum()}'
                                 for key, p in model.named_parameters()
                                 if p.grad is not None and (not torch.isfinite(torch.norm(p.grad.detach(), 2.0)).all())]))
    else:
        grad_norm = 0.0
    if torch.is_tensor(grad_norm):
        grad_norm = grad_norm.item()
    return grad_norm, is_overflow

def backprop_loss_(loss, optimizer, fp16_run=False, inputs=None, retain_graph=None):
    if fp16_run:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(inputs=inputs, retain_graph=retain_graph)
    else:
        loss.backward(inputs=inputs, retain_graph=retain_graph)

def optimizer_step_(optimizer, grad_norm):
    if math.isfinite(grad_norm):
        optimizer.step()

def complete_optimizer_step(loss, model: List[nnModule], optimizer, batch_size: int, weight_decay, learning_rate, grad_clip_thresh, fp16_run=False, inputs=None, retain_graph=None, skip_nan=True, test_run=False):
    backprop_loss_(loss, optimizer, fp16_run, inputs, retain_graph)
    
    w.print2if(   learning_rate == 0.0 and not test_run, 'learning_rate == 0.0')
    w.print2if(grad_clip_thresh == 0.0 and not test_run, 'grad_clip_thresh == 0.0')
    
    # adjust inputs with batch_size
    grad_clip_thresh = float(grad_clip_thresh) / batch_size**0.5
    learning_rate = float(learning_rate) * batch_size**0.5
    weight_decay = float(weight_decay) * batch_size**0.5
    
    if test_run:
        weight_decay = 0.0
        learning_rate = 0.0
    
    # apply parameters to optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = float(learning_rate)
        param_group['weight_decay'] = float(weight_decay)
    
    if skip_nan and not math.isfinite(loss):
        w.print2("Loss overflow.  Skipping optimizer step")
        return float('nan'), True
    
    if not (isinstance(model, tuple) or isinstance(model, list)):
        model: List[nnModule] = [model,]
    grad_norm, is_overflow = 0.0, False
    for m in model:
        _ = clip_grad_norm_(m, optimizer, grad_clip_thresh, fp16_run)
        grad_norm += (_[0] * batch_size**0.5)
        is_overflow = is_overflow or _[1]
        
        for k, p in m.named_parameters():
            #assert p is not None, f'{k} is None'
            #assert p.grad is not None, f'{k}.grad is None'
            if p.grad is not None and p.grad.detach().abs().sum() == 0:
                w.print2(f'{k}: has zero learning signal (gradients)')
    
    if not is_overflow:
        optimizer_step_(optimizer, grad_norm)
    return grad_norm, is_overflow


class MovingDictAverage(dict):
    """dict that tracks the numerical average of it's values in realtime"""
    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__(d)
        self.sum = sum(self.values())
    
    def __setitem__(self, key, item):
        if not hasattr(self, 'sum'):
            self.sum = sum(self.values())
        if key in self:
            self.sum -= self.__getitem__(key)
        o = super().__setitem__(key, item)
        self.sum += self.__getitem__(key)
        return o
    
    def update(self, new_d: Union[Dict, Tuple[str, Any]]):
        if not hasattr(self, 'sum'):
            self.sum = sum(self.values())
        if isinstance(new_d, dict):
            new_d_items = new_d.items()
        else:
            new_d_items = new_d
        for k, v in new_d_items:
            if k in self:
                self.sum -= self[k]
        o = super().update(new_d)
        for k, v in new_d_items:
            self.sum += v
        return o
    
    def pop(self, *args):
        if not hasattr(self, 'sum'):
            self.sum = sum(self.values())
        key, default = list(args) + [None]
        self.sum -= self[key]
        return super().pop(*args)
    
    def mean(self):
        if not hasattr(self, 'sum'):
            self.sum = sum(self.values())
        return self.sum / len(self)

class MovingPercentile:
    def __init__(self, percentile: float, max_len = None):
        assert 0.0 <= percentile <= 1.0, 'percentile must be between 0. and 1.'
        self.percentile = percentile
        self.max_len = max_len
        
        self.a_max = None
        self.a_list = []
        self.b_min = None
        self.b_list = []
        
        self.all_list = []
    
    def state_dict(self):
        return {
            'a_max': self.a_max,
            'a_list': self.a_list,
            'b_min': self.b_min,
            'b_list': self.b_list,
            'all_list': self.all_list,
        }
    
    def load_state_dict(self, d):
        self.a_max = d['a_max']
        self.a_list = d['a_list']
        self.b_min = d['b_min']
        self.b_list = d['b_list']
        self.all_list = d['all_list']
    
    def __call__(self, new_value):
        if self.max_len is not None:
            self.all_list.append(new_value)
        a_list_max = max(self.a_list) if self.a_list else -float('inf')
        if new_value <= a_list_max:
            self.a_list.append(new_value)
        else:
            self.b_list.append(new_value)
        
        if self.max_len is not None and len(self.all_list) > self.max_len: # if max_len and all_list is longer than max_len
            oldest_val = self.all_list.pop(0) # remove oldest entry from all_list and a_list/b_list
            try:
                a_list_old_index = self.a_list.index(oldest_val)
            except ValueError as ex:
                a_list_old_index = None
            if a_list_old_index is not None:
                self.a_list.pop(a_list_old_index)
            else:
                self.b_list.pop(self.b_list.index(oldest_val))
        
        a_len = len(self.a_list)
        total_len = a_len + len(self.b_list)
        a_max = None
        b_min = None
        if a_len / total_len > self.percentile: # if a_list (under percentile) is too large
            # move max element from a_list to b_list
            a_max = a_max if a_max is not None else max(self.a_list)
            self.b_list.append(self.a_list.pop(self.a_list.index(a_max)))
            b_min = a_max
            a_max = None
        elif a_len / total_len < self.percentile:# elif b_list (over percentile) is too larger
            # move min element from b_list to a_list
            b_min = b_min if b_min is not None else min(self.b_list)
            self.a_list.append(self.b_list.pop(self.b_list.index(b_min)))
            a_max = b_min
            b_min = None
        
        if len(self.b_list) > len(self.a_list):
            quantile = b_min if b_min is not None else min(self.b_list)
        else:
            quantile = a_max if a_max is not None else max(self.a_list)
        return quantile

class OptimizerModule:
    """
    Used during training.  \n
    1) Takes model  \n
    2) Creates Optimizer(s) for model parameters.  \n
    3) Can be called with module.step(loss_g) to update model weights  \n
    4) Can be called with module.update_config() to update optimizer config without rebuilding optimizer
    """
    def __init__(self, optimizer_config, model=None):
        # init params
        self.source      = optimizer_config['source']
        self.gen_type    = optimizer_config['gen_type']
        self.dis_type    = optimizer_config['dis_type']
        self.set_to_none = bool(optimizer_config['set_to_none'])
        self.amp_level   = int(optimizer_config['amp_level'])
        
        # live params
        assert 'grad_clip_thresh' not in optimizer_config, "'grad_clip_thresh' is depreciated, please replace with grad_clip_quantile or remove grad_clip_thresh"
        self.learning_rate   : float = optimizer_config.get('learning_rate', 0.0)
        self.weight_decay    : float = optimizer_config.get('weight_decay', 0.0)
        
        self.grad_clip_quantile: float = optimizer_config.get('grad_clip_quantile', 0.1)
        self.grad_g_tracker: MovingPercentile = None
        self.grad_d_tracker: MovingPercentile = None
        self.reset_grad_trackers()
        self.grad_g_clip_thresh: float = float('inf')
        self.grad_d_clip_thresh: float = float('inf')
        
        # maybe params
        self.optimizer_g: torch.optim.optimizer = None
        self.optimizer_d: torch.optim.optimizer = None
        
        # maybe init optimizer
        if model is not None:
            self.init_optimizers(model)
    
    def reset_grad_trackers(self):
        # gradient tracking (for calculating percentile based grad clipping - see AutoClip paper)
        self.grad_g_tracker = MovingPercentile(self.grad_clip_quantile, max_len=2048)
        self.grad_d_tracker = MovingPercentile(self.grad_clip_quantile, max_len=2048)
    
    def update_config(self, optimizer_config):
        if 'learning_rate' in optimizer_config:
            self.learning_rate = float(optimizer_config['learning_rate'])
        if 'grad_clip_quantile' in optimizer_config:
            self.grad_clip_quantile: float = optimizer_config['grad_clip_quantile']
        if 'weight_decay' in optimizer_config:
            self.weight_decay = float(optimizer_config['weight_decay'])
    
    def init_optimizers(self, model):
        kwargs = {'lr': 0.0, 'weight_decay': 0.0}
        
        if hasattr(model, 'generator'):
            g_weights = filter(lambda p: p.requires_grad, model.generator.parameters())
            self.optimizer_g = get_optimizer(self.source, self.gen_type)(g_weights, **kwargs)
        
        if hasattr(model, 'discriminator'):
            d_weights = filter(lambda p: p.requires_grad, model.discriminator.parameters())
            self.optimizer_d = get_optimizer(self.source, self.dis_type)(d_weights, **kwargs)
        
        w.print1(f"Optimizing {sum(p.numel() for p in model.parameters() if p.requires_grad):,} Parameters")
    
    def save(self, opt_path):
        d = {}
        d['optimizer_type'] = [self.source, self.gen_type]
        d['optimizer'] = self.optimizer_g.state_dict()
        d['grad_g_tracker'] = self.grad_g_tracker.state_dict()
        d['grad_d_tracker'] = self.grad_d_tracker.state_dict()
        if hasattr(self, 'optimizer_d') and self.optimizer_d is not None:
            d['optimizer_d'] = self.optimizer_d.state_dict()
        if amp is not None and self.amp_level != 0:# skip if amp_level is 0
            d['amp_statedict'] = amp.state_dict()
        safe_write(d, opt_path)
        w.print1(f'Saved OptimizerModule to "{opt_path}"')
     
    def load(self, opt_path):
        d = torch.load(opt_path, map_location='cpu')
        if d.get('optimizer_type', []) != [self.source, self.gen_type]:
            return
        try:
            self.optimizer_g.load_state_dict(d['optimizer'])
            if 'optimizer_d' in d:
                self.optimizer_d.load_state_dict(d['optimizer_d'])
        except ValueError as ex:
            w.print2(str(ex))
        if 'amp_statedict' in d and amp is not None:
            try:
                amp.load_state_dict(d['amp_statedict'])
            except Exception as ex:
                traceback.print_exc()
        if 'grad_g_tracker' in d:
            self.grad_g_tracker.load_state_dict(d['grad_g_tracker'])
        if 'grad_d_tracker' in d:
            self.grad_d_tracker.load_state_dict(d['grad_d_tracker'])
        w.print1(f'Loaded OptimizerModule from "{opt_path}"')
    
    def _zero_param_grads(self):
        kwargs = {} if self.source == 'apex' else {'set_to_none': self.set_to_none}
        self.optimizer_g.zero_grad(**kwargs)
        if hasattr(self, 'optimizer_d') and self.optimizer_d is not None:
            self.optimizer_d.zero_grad(**kwargs)
    
    def step(self, model, loss_g, loss_d, batch_size, collate_grads=False, test_run=False, learning_rate=None, weight_decay=None):
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if weight_decay is not None:
            self.weight_decay = float(weight_decay)
        
        grad_norm_total = 0.0
        is_overflow_total = False
        
        grad_norm_g = None
        if hasattr(self, 'optimizer_g') and loss_g is not None:
            if collate_grads:
                backprop_loss_(loss_g, model.generator, self.amp_level > 0)
            else:
                grad_norm_g, is_overflow_g = complete_optimizer_step(
                    loss_g, model.generator, self.optimizer_g,
                    batch_size, self.weight_decay, self.learning_rate, self.grad_g_clip_thresh,
                    fp16_run=self.amp_level > 0, skip_nan=True, test_run=test_run)
                grad_norm_total += grad_norm_g
                is_overflow_total += is_overflow_g
                if self.grad_g_tracker is not None and math.isfinite(grad_norm_g):
                    self.grad_g_clip_thresh = self.grad_g_tracker(grad_norm_g)
                self._zero_param_grads()
        
        grad_norm_d = None
        if hasattr(self, 'optimizer_d') and loss_d is not None:
            if collate_grads:
                backprop_loss_(loss_d, model.discriminator, self.amp_level > 0)
            else:
                grad_norm_d, is_overflow_d = complete_optimizer_step(
                    loss_d, model.discriminator, self.optimizer_d,
                    batch_size, self.weight_decay, self.learning_rate, self.grad_d_clip_thresh,
                    fp16_run=self.amp_level > 0, skip_nan=True, test_run=test_run)
                grad_norm_total += grad_norm_d
                is_overflow_total += is_overflow_d
                if self.grad_d_tracker is not None and math.isfinite(grad_norm_d):
                    self.grad_d_clip_thresh = self.grad_d_tracker(grad_norm_d)
                self._zero_param_grads()
        
        return grad_norm_total, grad_norm_g, grad_norm_d

if __name__ == '__main__':
    # test OptimizerModule
    import os
    from copy import deepcopy
    import tempfile
    tmpdir = tempfile.gettempdir()
    
    w.setLevel('info')
    
    class MockModelModule(nnModule):
        def __init__(self):
            super().__init__()
            self.generator = torch.nn.Linear(10, 10)
            self.discriminator = torch.nn.Linear(10, 10)
    model = MockModelModule()
    optimizer_config = {
        "source": 'pytorch',
        "gen_type": 'adam',
        "dis_type": 'adam',
        "set_to_none": True,
        "amp_level": 0,
    }
    optm = OptimizerModule(optimizer_config, model=model)
    
    # check optimizer works
    if True:
        orig_state_dict = deepcopy(model.state_dict())
        orig_gen_state_dict = deepcopy(model.generator.state_dict())
        loss_g = model.generator(torch.randn(2, 10))
        batch_size = 2
        optm.step(model, loss_g.mean(), None, batch_size, learning_rate=1e-3, grad_clip_thresh=999.9)
        
        # should fail since discriminator is not updated but checking for everything being updated
        try:
            assert all(v1.ne(v2).any() for (k1, v1), (k2, v2) in zip(model.state_dict().items(), orig_state_dict.items()))
            raise NotImplementedError
        except AssertionError:
            pass
        except NotImplementedError:
            raise AssertionError
        # should succeed since only generator is checked
        assert all(v1.ne(v2).any() for (k1, v1), (k2, v2) in zip(model.generator.state_dict().items(), orig_gen_state_dict.items()))
    
    # check optimizer keeps parameters separated correctly
    if True:
        # both modules are ran, both should be updated
        orig_state_dict = deepcopy(model.state_dict())
        loss_g = model.generator(torch.randn(2, 10))
        loss_d = model.discriminator(torch.randn(2, 10))
        batch_size = 2
        optm.step(model, loss_g.mean(), loss_d.mean(), batch_size, learning_rate=1e-3, grad_clip_thresh=999.9)
        assert all(v1.ne(v2).any() for (k1, v1), (k2, v2) in zip(model.state_dict().items(), orig_state_dict.items()))
        
        # but when the losses are swapped, loss_g shouldn't be able to update optimizer_d and vice-versa,
        orig_state_dict = deepcopy(model.state_dict())
        loss_g = model.generator(torch.randn(2, 10))
        loss_d = model.discriminator(torch.randn(2, 10))
        batch_size = 2
        optm.step(model, loss_d.mean(), loss_g.mean(), batch_size, learning_rate=1e-3, grad_clip_thresh=999.9)
        
        assert all(v1.eq(v2).any() for (k1, v1), (k2, v2) in zip(model.state_dict().items(), orig_state_dict.items()))
    
    orig_optm_d_state_dict = deepcopy(optm.optimizer_d.state_dict())
    orig_optm_g_state_dict = deepcopy(optm.optimizer_g.state_dict())
    optm.save(os.path.join(tmpdir, 'optm.pt'))
    optm.load(os.path.join(tmpdir, 'optm.pt'))
    os.remove(os.path.join(tmpdir, 'optm.pt'))
    
    w.print1('OptimizerModule Test Completed!')