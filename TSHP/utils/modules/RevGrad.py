##########################################################################################################
# taken from https://github.com/janfreyberg/pytorch-revgrad and modified so alpha_ scales grad magnitude #
##########################################################################################################
from torch.nn import Module
from torch import tensor
from torch.autograd import Function


class scalegrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output
    
    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * alpha_
        return grad_input, None


scalegrad_func = scalegrad.apply


class ScaleGrad(Module):
    def __init__(self, alpha=1.):
        """
        A gradient scaling layer.
        This layer has no parameters, and simply scales the gradient
        in the backward pass.
        """
        super().__init__()
        
        self._alpha = tensor(alpha, requires_grad=False)
    
    def forward(self, input_):
        return scalegrad_func(input_, self._alpha)


class RevGrad(ScaleGrad):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(-alpha, *args, **kwargs)