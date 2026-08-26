# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
from packaging import version

from megatron.core.jit import jit_fuser
from megatron.core.utils import null_decorator, nvtx_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = bool(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


@jit_fuser
def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    """Computes the gradient for the SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed using the
            chain rule and the derivative of the SiLU activation function.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    """Computes the gradient for the biased SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


# Same tiling as torch.compile's triton_poi_fused_mul_silu_split_0:
#   xindex flattens [M, H], XBLOCK=1024, col = xindex % H, row = xindex // H.
# Extra: per-row absmax of the stored (rounded) y. Requires XBLOCK % H == 0 so
# each program owns complete rows and amax is a local reshape+max, no atomics.
@triton.jit
def _weighted_swiglu_row_amax_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    amax_ptr,
    xnumel,
    M,
    H: tl.constexpr,
    TWO_H: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % H
    x1 = xindex // H
    x2 = xindex
    gate = tl.load(x_ptr + (x0 + TWO_H * x1), xmask).to(tl.float32)
    up = tl.load(x_ptr + (H + x0 + TWO_H * x1), xmask).to(tl.float32)
    prob = tl.load(w_ptr + x1, xmask, eviction_policy="evict_last").to(tl.float32)
    y = gate * tl.sigmoid(gate) * up * prob
    y_store = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + x2, y_store, xmask)

    nrows: tl.constexpr = XBLOCK // H
    absv = tl.where(xmask, tl.abs(y_store.to(tl.float32)), 0.0)
    row_max = tl.max(absv.reshape(nrows, H), axis=1)
    rows = (xoffset // H) + tl.arange(0, nrows)
    tl.store(amax_ptr + rows, row_max, mask=rows < M)


@triton.jit
def _weighted_swiglu_row_amax_tiled_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    amax_ptr,
    M,
    H,
    TWO_H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fallback when no power-of-two XBLOCK owns complete rows (odd H)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for col0 in tl.range(0, H, BLOCK_N):
        cols = col0 + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < H)
        gate = tl.load(
            x_ptr + rows[:, None] * TWO_H + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            x_ptr + rows[:, None] * TWO_H + H + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        prob = tl.load(
            w_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        y_store = (gate * tl.sigmoid(gate) * up * prob[:, None]).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + rows[:, None] * H + cols[None, :], y_store, mask=mask)
        acc = tl.maximum(
            acc, tl.max(tl.where(mask, tl.abs(y_store.to(tl.float32)), 0.0), axis=1)
        )
    tl.store(amax_ptr + rows, acc, mask=row_mask)


def _pick_xblock(h: int):
    """Inductor uses XBLOCK=1024; keep that when it maps to whole rows."""
    for xb in (1024, 2048, 512, 256, 128, 64, 32):
        if xb % h == 0:
            return xb
    return None


def _weighted_swiglu_with_row_amax_triton(input, weights):
    """Fused ``silu(gate) * up * weights`` plus per-row abs-max of stored ``y``."""
    assert input.is_cuda and input.dim() == 2
    m, two_h = input.shape
    assert two_h % 2 == 0, "SwiGLU input last dim must be even"
    h = two_h // 2
    x = input.contiguous()
    w = weights.reshape(-1).contiguous()
    assert w.numel() == m, f"weights length {w.numel()} != rows {m}"
    y = torch.empty((m, h), device=x.device, dtype=x.dtype)
    row_amax = torch.empty((m,), device=x.device, dtype=torch.float32)
    if m == 0 or h == 0:
        return y, row_amax
    xnumel = m * h
    xblock = _pick_xblock(h)
    if xblock is not None:
        _weighted_swiglu_row_amax_kernel[(triton.cdiv(xnumel, xblock),)](
            x,
            w,
            y,
            row_amax,
            xnumel,
            m,
            H=h,
            TWO_H=two_h,
            XBLOCK=xblock,
            num_warps=4,
            num_stages=1,
        )
    else:
        block_n = min(1024, triton.next_power_of_2(h))
        block_m = max(1, 1024 // block_n)
        _weighted_swiglu_row_amax_tiled_kernel[(triton.cdiv(m, block_m),)](
            x,
            w,
            y,
            row_amax,
            m,
            h,
            two_h,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
    return y, row_amax


class BiasSwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            bias (torch.Tensor): Bias tensor to be added to input before SwiGLU.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying bias addition followed by SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - Gradient with respect to the bias tensor
                - None for fp8_input_store parameter
        """
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - None for fp8_input_store parameter
        """
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = swiglu_back(grad_output, input)
        return tmp, None, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased SwiGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SwiGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SwiGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SwiGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SwiGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_swiglu_row_amax_available() -> bool:
    """True when the Triton weighted-SwiGLU + row-amax kernel can run."""
    return bool(HAVE_TRITON) and torch.cuda.is_available()


def te_swiglu_with_row_amax_available() -> bool:
    """Backward-compatible alias for ``weighted_swiglu_row_amax_available``."""
    return weighted_swiglu_row_amax_available()


class WeightedSwiGLUWithRowAmaxFunction(torch.autograd.Function):
    """Weighted SwiGLU that also emits a detached per-token row amax.

    Forward matches ``weighted_swiglu`` (silu(gate)*up*probs) in one Triton
    kernel and reduces ``max_j |y[i, j]|`` in the same pass. Backward matches
    ``WeightedSwiGLUFunction``; amax is not differentiated.
    """

    @staticmethod
    def forward(ctx, input, weights, fp8_input_store):
        if not weighted_swiglu_row_amax_available():
            raise RuntimeError("Triton weighted SwiGLU + row-amax kernel is not available")
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        y_out, row_amax = _weighted_swiglu_with_row_amax_triton(input, weights)
        return y_out, row_amax

    @staticmethod
    def backward(ctx, grad_output, grad_row_amax):
        del grad_row_amax  # Quantize amax is not in the autograd graph.
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def weighted_swiglu_with_row_amax_impl(input, bias, weights, fp8_input_store=False):
    """Weighted SwiGLU plus per-token row amax for NVFP4 fc2 skip-K1.

    Returns ``(y, row_amax)`` where ``y`` matches ``weighted_bias_swiglu_impl``
    and ``row_amax`` is fp32 ``[N]`` equal to ``(y.abs().amax(-1)`` after
    ``y *= |probs|``).
    """
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu + row-amax fusion")
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    y, row_amax = WeightedSwiGLUWithRowAmaxFunction.apply(input, weights, fp8_input_store)
    if len(ori_shape) != 2:
        y = y.view(ori_shape[0], ori_shape[1], -1)
    return y, row_amax


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
