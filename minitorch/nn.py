from typing import Tuple

from . import operators
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
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
        kh,  # Height of each tile
        new_width,  # Number of horizontal tiles
        kw,  # Width of each tile
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
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
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


max_op = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return a tensor with 1.0 in the position of the maximum value along dimension dim.
    The output tensor keeps the reduced dimension as size 1.
    """
    # Get max values using fast_max
    max_vals = max_op(input, dim)

    # Create mask for positions equal to max
    is_max = input == max_vals

    # Create a tensor that's 1 for the first position and 0 elsewhere
    shape = [1] * len(input.shape)
    shape[dim] = input.shape[dim]
    mask = input.zeros(shape)
    mask = mask + 1.0  # Make the first position 1

    # Keep only the first occurrence of max
    one_hot = (is_max * mask).float()  # type: ignore

    # Reshape to keep the reduced dimension as size 1
    out_shape = list(input.shape)
    out_shape[dim] = 1
    return one_hot.view(*out_shape)


def max(input: Tensor, dim: int) -> Tensor:
    """Return the maximum value along dimension dim.
    The output tensor keeps the reduced dimension as size 1.
    """
    # Get max values using fast_max
    max_vals = max_op(input, dim)

    # Create output shape with dim reduced to size 1
    out_shape = list(input.shape)
    out_shape[dim] = 1

    # Reshape to keep the reduced dimension as size 1
    return max_vals.view(*out_shape)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D
    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns
    -------
        Tensor : batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Use tile function to reshape input
    tiled, new_height, new_width = tile(input, kernel)

    # Apply max to the last dimension (contains the pooling window)
    pooled = max(tiled, -1)

    # Reshape back to 4D
    return pooled.view(batch, channel, new_height, new_width)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    # Subtract max for numerical stability
    max_vals = max(input, dim)
    shifted = input - max_vals

    # Compute exp
    exp_vals = shifted.exp()

    # Sum along dimension and reshape to match input
    sum_exp = exp_vals.sum(dim)
    out_shape = list(input.shape)
    out_shape[dim] = 1
    sum_exp = sum_exp.view(*out_shape)

    # Normalize
    return exp_vals / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply logsoftmax

    Returns:
    -------
        log softmax tensor

    """
    # Get max for numerical stability
    max_vals = max(input, dim)

    # Shift values
    shifted = input - max_vals

    # Compute exp and sum
    exp_vals = shifted.exp()
    sum_exp = exp_vals.sum(dim)
    out_shape = list(input.shape)
    out_shape[dim] = 1
    sum_exp = sum_exp.view(*out_shape)

    # Log-sum-exp trick: log(sum(exp(x - max))) + max
    return shifted - sum_exp.log()


def dropout(input: Tensor, rate: float = 0.5, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : dropout rate (0 = no dropout, 1 = drop all)
        ignore : if True, turns off dropout and returns input unchanged

    Returns:
    -------
        Tensor with random positions dropped to 0

    """
    # If ignore is True or rate is 0, return input unchanged
    if ignore or rate == 0:
        return input

    # If rate is 1, return zeros
    if rate == 1:
        return input * 0.0

    # Generate random mask
    rand_tensor = rand(input.shape)
    # Create mask using comparison operator directly
    mask = rand_tensor > rate

    # Scale output to maintain expected values
    scale = 1.0 / (1.0 - rate)

    return input * mask * scale


# class Max(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
#         """Forward pass for max operation."""
#         dim = int(dim.item())  # type: ignore
#         # Save input and dim for backward
#         ctx.save_for_backward(input, dim)

#         # Get max values
#         vals = input.f.max_reduce(input, dim)

#         # Create output shape with dim reduced to size 1
#         out_shape = list(input.shape)
#         out_shape[dim] = 1
#         return vals.view(*out_shape)

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
#         """Backward pass for max operation."""
#         input, dim = ctx.saved_values

#         # Get max values (keeping reduced dimension)
#         max_vals = input.f.max_reduce(input, dim)
#         out_shape = list(input.shape)
#         out_shape[dim] = 1
#         max_vals = max_vals.view(*out_shape)

#         # Create mask for positions equal to max
#         is_max = input == max_vals

#         # Create a tensor that's 1 for the first max position only
#         shape = [1] * len(input.shape)
#         shape[dim] = input.shape[dim]
#         first_pos = input.zeros(shape)
#         first_pos = first_pos + 1.0  # Make the first position 1

#         # Only keep the first occurrence of each max
#         grad_mask = is_max * first_pos

#         # Expand grad_output for broadcasting
#         for _ in range(len(input.shape) - len(grad_output.shape)):
#             grad_output = grad_output.unsqueeze(-1)

#         return grad_output * grad_mask, 0.0
