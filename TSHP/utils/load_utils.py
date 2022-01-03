import os

import torch
from torch.nn import functional as F

from TSHP.utils.saving.utils import safe_write


def load_state_dict_force(model, checkpoint_state_dict):
    changed_shape = False
    nmst = model.state_dict()  # new model state_dict, checkpoint model state_dict
    if len(checkpoint_state_dict.keys() ^ nmst.keys()) or any(checkpoint_state_dict[k].shape != nmst[k].shape for k, v in checkpoint_state_dict.items()):
        changed_shape = True
        print("Layers in checkpoint don't match current model. Some layer may be resized or reset. Expect a small drop in performance of the model.")
        print("\n".join({'ignored: ' + str(k) for k, v in sorted(checkpoint_state_dict.items()) if k not in nmst}))
        print("\n".join({'missing: ' + str(k) for k, v in sorted(nmst.items()) if k not in checkpoint_state_dict}))
        print("\n".join({'resize : ' + str(k) for k, v in sorted(checkpoint_state_dict.items()) if k in nmst and k.endswith('.conv.weight') and checkpoint_state_dict[k].shape != nmst[k].shape}))
        print("\n".join({'reset  : ' + str(k) for k, v in sorted(checkpoint_state_dict.items()) if k in nmst and not k.endswith('.conv.weight') and checkpoint_state_dict[k].shape != nmst[k].shape}))
        
        # copy over layers that match
        nmst.update({k: v for k, v in checkpoint_state_dict.items() if k in nmst and checkpoint_state_dict[k].shape == nmst[k].shape})
        
        def resize_weights(w, new_size):
            if w.shape == new_size:
                return w
            init_dtype = w.dtype
            n_dim = w.dim()
            w_sum = w.sum(dtype=torch.float)
            modelookup = {1: 'linear',
                          2: 'bicubic',
                          3: 'trilinear', }
            w = F.interpolate(w.float()[None, None, ...], size=new_size, mode=modelookup[n_dim],
                              align_corners=True)
            w = w * (w_sum / w.sum(dtype=torch.float))  # ensure sum of weights remains unchanged
            w = w.squeeze(0).squeeze(0).to(init_dtype)
            return w
        
        # copy over layers that are the wrong size (interpolate/resize them to match)
        nmst.update({k: resize_weights(checkpoint_state_dict[k], nmst[k].shape) for k, v in checkpoint_state_dict.items() if
                     k in nmst and k.endswith('.conv.weight') and checkpoint_state_dict[k].shape != nmst[k].shape and checkpoint_state_dict[k].dim() == nmst[k].dim()})
        
        model.load_state_dict(nmst)
    else:
        model.load_state_dict(checkpoint_state_dict, strict=True)
    return changed_shape


# shared weights module
class SharedWModule:
    def __init__(self, shared_weights_config):
        self.shared_weights = shared_weights_config
        self.cp_iters = {}
        self.model_cp_iter = {}
    
    def get_lookup_attr(self, obj, attr):
        keydict = getattr(obj, attr)
        if type(keydict) in [list, tuple]:
            if type(keydict[0]) in [list, tuple]:
                keydict = {x[0]: x[1] for x in keydict}
            else:
                keydict = {i: x for i, x in enumerate(keydict)}
        elif type(keydict) is dict:
            pass
        else:
            raise NotImplementedError
        return keydict
    
    def get_next_free_id(self, d):
        new_id = 0
        ids = [v for k, v in d.items()]
        while new_id in ids:# keep increasing new_id till a free value is found
            new_id += 1
        return new_id
    
    def save_weight(self, index, model, statedict_key, dictlookup_key, embed_path, embed_dim, freeze_embed):
        state_dict = model.state_dict()
        # extract text embed (and keydict)
        try:
            curr_embed = next(v for k, v in state_dict.items() if k.endswith(statedict_key))
        except StopIteration:
            print(f"{statedict_key} not found. Skipping.")
            return# if there is no k in state_dict that ends with statedict_key, just exit the function.
        keydict = self.get_lookup_attr(model, dictlookup_key)# {key: internal_id ...}
        
        if os.path.exists(embed_path):# load current embed
            saved_embed, saved_iters, saved_keydict = torch.load(embed_path, map_location='cpu')
            
            if saved_embed.shape[1] < embed_dim:
                saved_embed = torch.cat((
                    saved_embed,
                    torch.randn(saved_embed.shape[0], embed_dim-saved_embed.shape[1]) * 1e-4
                ), dim=1)
            
            # mix with updated version
            for key, key_id in keydict.items():
                if key not in saved_keydict.keys():  # if speaker is in model but has never been added to saved embed before
                    saved_keydict[key] = self.get_next_free_id(saved_keydict) # add speaker to saved_keydict
                    if saved_embed.shape[0] <= saved_keydict[key]:
                        saved_embed = torch.cat((saved_embed, saved_embed[0:1]), dim=0)
                        new_v = torch.randn(embed_dim, device=saved_embed.device, dtype=saved_embed.dtype) * curr_embed.data.std().to(saved_embed)
                        saved_embed[saved_keydict[key]] = new_v 
                min_dim = min(curr_embed.shape[1], saved_embed.shape[1])
                saved_embed[saved_keydict[key], :min_dim] = curr_embed.data[keydict[key], :min_dim].to(saved_embed)
            
            # update saved_iters
            passed_iters = model.iteration - self.model_cp_iter[index]  # iterations passed since the embedding was last loaded
            for k, v in saved_iters.items():
                if k in keydict.keys():# if key in current model, increase iter count for that key by the amount of model iters
                    saved_iters[k] = saved_iters[k] + passed_iters
            
            # save to file (and wait if another process is currently accessing this file)
            safe_write([saved_embed, saved_iters, saved_keydict], embed_path)
        else:
            # save to file (and wait if another process is currently accessing this file)
            safe_write([curr_embed, {k: model.iteration for k in keydict.keys()}, keydict], embed_path)
    
    def save_weights(self, model):
        for index, d in enumerate(self.shared_weights):
            self.save_weight(index, model, **d)
    
    def load_weight(self, index, model, statedict_key, dictlookup_key, embed_path, embed_dim, freeze_embed):
        # save current iters
        self.model_cp_iter[index] = model.iteration
        self.cp_iters[index] = model.iteration
        if not os.path.exists(embed_path):
            print(f'Shared Weights at "{embed_path}" doesn\'t exist. Skipping the load_weight()')
            return
        if not any(k.endswith(statedict_key) for k in model.state_dict().keys()):
            print(f'Model does not contain {statedict_key}, Skipping the load_weight()')
            return
        
        state_dict = model.state_dict()
        # extract text embed (and keydict)
        curr_embed = next(v for k, v in state_dict.items() if k.endswith(statedict_key))
        keydict = self.get_lookup_attr(model, dictlookup_key)
        
        # load current embed
        saved_embed, saved_iters, saved_keydict = torch.load(embed_path, map_location='cpu')
        print(f"{statedict_key} loaded from shared file.")
        self.cp_iters[index] = saved_iters
        
        # mix with updated version
        for key, key_id in keydict.items():# dict of {keyname: internal_id}
            if key not in saved_keydict.keys():
                continue  # don't try to load speakers that are not in the saved list
            min_dim = min(curr_embed.shape[1], saved_embed.shape[1])
            curr_embed.data[key_id, :min_dim] = saved_embed[saved_keydict[key], :min_dim]
        
        # maybe freeze embed
        curr_embed.requires_grad = not freeze_embed
    
    def load_weights(self, model):
        for index, d in enumerate(self.shared_weights):
            self.load_weight(index, model, **d)