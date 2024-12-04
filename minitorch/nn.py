from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw
    
    # Make input contiguous before view operation
    input = input.contiguous()
    
    # Create a view of the input tensor with the tiled structure
    tiled = input.view(
        batch, 
        channel,
        new_height,  # Number of vertical tiles
        kh,         # Height of each tile
        new_width,  # Number of horizontal tiles
        kw          # Width of each tile
    )
    
    # Permute to get the kernel dimensions at the end
    tiled = tiled.permute(0, 1, 2, 4, 3, 5)
    
    # Make contiguous again after permute
    tiled = tiled.contiguous()
    
    # Combine the kernel dimensions
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)
    
    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    
    # Use the tile function to reshape
    tiled, new_height, new_width = tile(input, kernel)
    
    # Average over the last dimension (which contains the kernel values)
    pooled = tiled.mean(4)
    
    # Ensure correct output shape
    return pooled.view(batch, channel, new_height, new_width)