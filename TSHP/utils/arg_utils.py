import types

def get_args(*func, kwargs=True):
    args = list()
    if type(func) in [list, tuple]:
        for f in func:
            args.extend(f.__code__.co_varnames[:f.__code__.co_argcount])
            if kwargs:
                args.extend(f.__code__.co_varnames[f.__code__.co_argcount:f.__code__.co_kwonlyargcount])
    elif isinstance(func, types.ClassType):
        args.extend(func.__call__.__code__.co_varnames[:func.__call__.__code__.co_argcount])
        if kwargs:
            args.extend(func.__call__.__code__.co_varnames[func.__call__.__code__.co_argcount:func.__call__.__code__.co_kwonlyargcount])
    elif isinstance(func, types.FunctionType):
        args.extend(func.__code__.co_varnames[:func.__code__.co_argcount])
        if kwargs:
            args.extend(func.__code__.co_varnames[func.__code__.co_argcount:func.__code__.co_kwonlyargcount])
    elif isinstance(func, types.CodeType):
        args.extend(func.co_varnames[:func.co_argcount])
        if kwargs:
            args.extend(func.co_varnames[func.co_argcount:func.co_kwonlyargcount])
    args = tuple(set(args))
    return args

def replace_args(args, replace_dict=None):
    if replace_dict is None:
        replace_dict = {}
    new_args = list()
    for arg in args:
        new_arg = replace_dict[arg] if arg in replace_dict.keys() else arg
        new_args.append(new_arg)
    return tuple(set(new_args))

def rename_argdict(argdict, replace_dict=None, invert=True):
    if replace_dict is None:
        replace_dict = {}
    if invert:
        replace_dict = {v: k for k, v in replace_dict.items()}
    argdict = {replace_dict[k] if k in replace_dict else k: v for k,v in argdict.items()}
    return argdict

def force_any(func, valid_kwargs=None, any_gt_pr=True, *args, **kwargs):
    return force(func, valid_kwargs=valid_kwargs, any_gt_pr=any_gt_pr, *args, **kwargs)

def force(func, valid_kwargs=None, any_gt_pr=False, *args, **kwargs):
    if valid_kwargs is True:
        return func(*args, **kwargs)
    elif valid_kwargs is None:
        valid_kwargs = get_args(func, kwargs=True)
    if not any_gt_pr:
        return func(*args, **{k: v for k, v in kwargs.items() if k in valid_kwargs})
    else:
        # (below is example with mel, any feature could be used)
        #  pr_mel -> any_mel
        # any_mel -> any_mel
        #  gt_mel -> any_mel
        # priority order for rename is [batch, any, gt] -> any
        # ideally any_mel should never be in the input, but if it does exist then try to replace with pr_mel
        kwargs_to_use = {}
        for p in ['pr_', 'any_', 'gt_']:
            for k, v in kwargs.items():
                any_k = k.replace(p, 'any_')
                if k.startswith(p) and any_k in valid_kwargs and any_k not in kwargs_to_use:
                    kwargs_to_use[any_k] = v
        kwargs_to_use.update({k: v for k, v in kwargs.items() if k in valid_kwargs and k not in kwargs_to_use})
        return func(*args, **kwargs_to_use)