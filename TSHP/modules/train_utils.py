import os

import torch

from TSHP.utils import warnings as w
 
def get_all_model_refs():
    refs = []
    models_dir = os.path.join(os.path.split(os.path.split(__file__)[0])[0], 'models')
    for modeltype_dirname in os.listdir(models_dir):
        modeltype_dir = os.path.join(models_dir, modeltype_dirname)
        if not os.path.isdir(modeltype_dir):
            continue
        
        for model_dirname in os.listdir(modeltype_dir):
            model_dir = os.path.join(modeltype_dir, model_dirname)
            if not os.path.isdir(model_dir):
                continue
            
            if 'model.py' in os.listdir(model_dir):
                refs.append(f'{modeltype_dirname}.{model_dirname}')
    return refs


def find_weight_path(base_directory, model_identity, run_name, weight_name):
    # check checkpoints directory exists
    assert 'checkpoints' in base_directory or 'runs' in base_directory, 'current directory is not checkpoints directory'
    w.print2if('default_config.yaml' not in os.listdir(base_directory), 'default_config.yaml missing from current directory')
    
    allowed_refs = get_all_model_refs()
    assert model_identity in allowed_refs, f'{model_identity} is not a valid model_identity.\nvalid model_identities are: {allowed_refs}'
    
    model_type, model_name = model_identity.split('.')
    
    # check model directory exists
    os.makedirs(os.path.join(base_directory, *model_identity.split('.')), exist_ok=True)
    assert model_type in os.listdir(base_directory)
    assert model_name in os.listdir(os.path.join(base_directory, model_type))
    
    # check run directory exists
    weight_path = None
    run_path = os.path.join(base_directory, *model_identity.split('.'), run_name)
    if run_name in os.listdir(os.path.join(base_directory, *model_identity.split('.'))):
        w.print1(f'run {run_name} found.')
        if weight_name is None:
            weight_path = None
        elif os.path.exists(weight_name):
            weight_path = weight_name
        elif os.path.exists(os.path.join(run_path, 'weights')) and weight_name+'.ptw' in os.listdir(os.path.join(run_path, 'weights')):
            weight_path = os.path.join(run_path, 'weights', weight_name+'.ptw')
        elif os.path.exists(os.path.join(run_path, 'weights')) and weight_name in os.listdir(os.path.join(run_path, 'weights')):
            weight_path = os.path.join(run_path, 'weights', weight_name)
        else:
            w.print2(f'{weight_name} not found in {run_name}/weights')
            weight_path = None
    else:  # if not exists: create from template
        w.print1(f'run {run_name} does not exist. Creating folder.')
        os.makedirs(run_path, exist_ok=True)
    w.print1(f'using weight_path: {weight_path}')
    return run_path, weight_path


def guess_model_ref_from_path(path, ref=None, allow_notnone_ref=True):
    assert allow_notnone_ref or ref is None, f'called guessing_model_ref_from_path() but model_ref is not None. got model_ref: {ref}'
    if ref is None:
        # try to guess model reference from server config file
        
        # try guess model reference from path
        for possible_ref in reversed(sorted(get_all_model_refs(), key=lambda x: len(x))):
            if possible_ref.lower() in path.lower().replace('/', '.').replace('\\', '.'):
                return possible_ref
        for possible_ref in reversed(sorted(get_all_model_refs(), key=lambda x: len(x))):
            if possible_ref.split(".")[-1].lower() in path.lower():
                return possible_ref
    
        # try to guess model reference from model config file
    
        # can't guess model reference, raise exception
        raise Exception(f"Could not guess model reference for path: {path}")
    return ref


def deepto(d, *args, **kwargs):
    if isinstance(d, dict):
        do = {}
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                do[k] = deepto(v, *args, **kwargs)
            elif torch.is_tensor(v):
                do[k] = v.to(*args, **kwargs)
            else:
                do[k] = v
        return do
    elif isinstance(d, (list, tuple)):
        do = []
        for v in d:
            if isinstance(v, (dict, list, tuple)):
                do.append(deepto(v, *args, **kwargs))
            elif torch.is_tensor(v):
                do.append(v.to(*args, **kwargs))
            else:
                do.append(v)
        return do
    elif type(d) is torch.Tensor:
        return d.to(*args, **kwargs)
    else:
        raise NotImplementedError