"""Numerical SAM/element-wise ASAM geometry on canonical optimizer storage.

These pure helpers are also used by the monitoring path. They do not own
optimizer state, choose a microbatch schedule, or advance an optimizer.
"""
import copy
import math


def displacement(blocks, gradients, config):
    metric = [[abs(float(w)) + float(config["eta"]) if config["adaptive"] else 1.0
               for w in block["weight"]] for block in blocks]
    dual_norm = math.hypot(*(a * g for block, factors, grad in zip(blocks, metric, gradients)
                             if block.get("trainable", True) for a, g in zip(factors, grad)))
    factor = float(config["radius"]) / (dual_norm + float(config["epsilon"]))
    return [[factor * a * a * g if block.get("trainable", True) else 0.0
             for a, g in zip(factors, grad)] for block, factors, grad in zip(blocks, metric, gradients)]


def perturbed(state, delta):
    result = copy.deepcopy(state)
    for block, change in zip(result["blocks"], delta):
        block["weight"] = [w + d for w, d in zip(block["weight"], change)]
    return result


def receipt(window_id, stats, gradients, blocks, delta, peak, overflow):
    return {"window_id": window_id, "base_total_loss": stats["total_loss"],
            "base_grad_norm": math.hypot(*(g for b, gs in zip(blocks, gradients)
                                          if b.get("trainable", True) for g in gs)),
            "perturbation_norm": math.hypot(*(d for group in delta for d in group)),
            "base_scaled_grad_peak": peak, "base_overflow": overflow}
