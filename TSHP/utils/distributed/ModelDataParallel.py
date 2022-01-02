import torch
import torch.distributed as dist
from torch.autograd import Variable

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

'''
Modifies existing model to do gradient allreduce, but doesn't change class
so you don't need "module"

Parameters are broadcasted to the other processes on initialization of DistributedDataParallel,
and will be allreduced at the finish of the backward pass.

Does not (currently) support different batch sizes for each GPU.
'''
class apply_gradient_allreduce:
    def __init__(self, weight=1.0):
        self.weights = torch.zeros(dist.get_world_size(), device='cuda')
        self.weights[dist.get_rank()] = weight
        dist.all_reduce(self.weights) # sync gradient weight between GPUs
        self.weight = self.weights[dist.get_rank()]
    
    def __call__(self, module):
        if not hasattr(dist, '_backend'):
            module.warn_on_half = True
        else:
            module.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False
        
        # broadcast model from GPU 0 to all GPUs so every GPU has an exact copy of the current model state
        for p in module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            dist.broadcast(p, 0)
        
        def allreduce_params():
            if(module.needs_reduction):
                module.needs_reduction = False
                buckets = {}
                for param in module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = param.data.dtype
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if module.warn_on_half:
                    if torch.cuda.HalfTensor in buckets:
                        print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                              " It is recommended to use the NCCL backend in this case. This currently requires" +
                              "PyTorch built from top of tree master.")
                        module.warn_on_half = False
                
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads) # flatten tensors for dist.all_reduce() compatibility
                    if self.weight != 1.0:
                        coalesced *= self.weight.to(coalesced) # multiply by weight factor (will do nothing unless batch sizes vary between GPUs)
                    dist.all_reduce(coalesced) # sum all 'coalesced' varibles from every GPU together inplace
                    coalesced /= self.weights.sum().to(coalesced) # divide by num_gpus to get average 'coalesced'
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced) # copy averaged variable back to original location
        
        # add backward hook to every parameter: replace_current_gradient_with_average_gradient_from_every_GPU()
        for param in list(module.parameters()):
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)
        
        def set_needs_reduction(self, input, output):
            self.needs_reduction = True
        
        module.register_forward_hook(set_needs_reduction)
        return module
