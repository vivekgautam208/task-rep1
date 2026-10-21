import math


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def softmax_with_lse(logits):
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    denom = math.fsum(exps)
    return [v / denom for v in exps], peak + math.log(denom)


def expert_layout(dim, hidden):
    w1 = hidden * dim
    b1 = hidden
    w2 = hidden
    return w1, b1, w2


def expert_forward(weight, x, hidden, dropout_p, mask_bits):
    dim = len(x)
    w1n, b1n, _ = expert_layout(dim, hidden)
    keep_prob = 1.0 - float(dropout_p)
    scale = 1.0 / keep_prob if keep_prob > 0.0 else 0.0
    acts = []
    dropped = []
    for h in range(hidden):
        base = h * dim
        a = math.fsum(weight[base + d] * x[d] for d in range(dim)) + weight[w1n + h]
        z = math.tanh(a)
        bit = bool(mask_bits[h])
        hd = z * scale if bit else 0.0
        acts.append((a, z, bit))
        dropped.append(hd)
    y = math.fsum(weight[w1n + b1n + h] * dropped[h] for h in range(hidden))
    return y, acts, dropped


def expert_backward(weight, x, hidden, dropout_p, mask_bits, upstream):
    dim = len(x)
    w1n, b1n, _ = expert_layout(dim, hidden)
    y, acts, dropped = expert_forward(weight, x, hidden, dropout_p, mask_bits)
    grad = [0.0] * len(weight)
    keep_prob = 1.0 - float(dropout_p)
    scale = 1.0 / keep_prob if keep_prob > 0.0 else 0.0
    for h in range(hidden):
        w2_idx = w1n + b1n + h
        grad[w2_idx] += upstream * dropped[h]
        _, z, bit = acts[h]
        if bit:
            da = upstream * weight[w2_idx] * scale * (1.0 - z * z)
            grad[w1n + h] += da
            base = h * dim
            for d in range(dim):
                grad[base + d] += da * x[d]
    return grad


def zero_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

FP16_MAX = 65504.0

def amp_scaled_status(values, scale):
    peak = 0.0
    overflow = False
    for value in values:
        scaled = abs(float(value) * float(scale))
        if math.isfinite(scaled):
            peak = max(peak, scaled)
            if scaled > FP16_MAX:
                overflow = True
        else:
            peak = float("inf")
            overflow = True
    return peak, overflow

def amp_microbatch_status(raw_local_gradient_values, scale):
    """Return AMP status for one microbatch backward.

    Input is the raw task + z-loss gradient produced by that backward, before
    window-token-weight normalization.  The window-global auxiliary router
    gradient is not a microbatch-local value and must not be included here.
    """
    return amp_scaled_status(raw_local_gradient_values, scale)

def amp_window_status(local_statuses, final_values, scale):
    """Combine microbatch statuses with the final normalized window gradient.

    ``final_values`` includes every window-global contribution, including the
    auxiliary router gradient.
    """
    final_peak, final_overflow = amp_scaled_status(final_values, scale)
    local_peak = max((float(peak) for peak, _ in local_statuses), default=0.0)
    local_overflow = any(bool(flag) for _, flag in local_statuses)
    return max(local_peak, final_peak), local_overflow or final_overflow

