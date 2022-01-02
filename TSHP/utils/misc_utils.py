import time
import warnings
from itertools import zip_longest

def zip_equal(*iterables):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError(f'Iterables have different lengths, got {[len(x) for x in iterables]} lens')
        yield combo


def deepupdate_dicts(a, b, main_call=True, warn_if_key_not_exists=False):  # b replaces anything from a that can't be merged
    if not main_call and (type(a) is not dict or type(b) is not dict):
        return b
    
    for k in a.keys():  # if something in b and a, call self and update them
        if k in b.keys():
            a[k] = deepupdate_dicts(a[k], b[k], main_call=False, warn_if_key_not_exists=warn_if_key_not_exists)
    
    for k in b.keys():  # if something in b and not a, copy from b to a
        if k not in a.keys():
            if warn_if_key_not_exists:
                warnings.warn(f'key: {k} does not exist in base dict.')
                time.sleep(0.5)
            a[k] = b[k]
    return a


def create_nested_dict(d):
    if not type(d) is type({}):
        return d
    d_out = {}
    key_set = sorted(list(set([k.split(".")[0] for k in d.keys()])))
    for i, d_ in enumerate([{k: v for k, v in d.items() if key == k.split(".")[0]} for key in key_set]):
        init_key = key_set[i]
        if any('.' in k for k in d_.keys()):
            d_out[init_key] = {'.'.join(k.split(".")[1:]): v for k, v in d_.items()}
            d_out[init_key] = create_nested_dict(d_out[init_key])
        else:
            for k, v in d_.items():
                d_out[init_key] = v
    return d_out