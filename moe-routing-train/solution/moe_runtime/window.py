"""Window loss aggregation and explicit-vector differentiation."""
import copy
import math
from . import network, objective
from .ops import amp_microbatch_status
from .storage import zeros, trainable_values


def public_forward(fwd):
    return {k: copy.deepcopy(v) for k, v in fwd.items() if not k.startswith("_")}


def evaluate(state, cfg, window, forwards=None):
    if forwards is None:
        forwards = [network.forward(state, cfg, str(window["id"]), mb) for mb in window["microbatches"]]
    W, loads, layers = network.window_statistics(state["model"], forwards)
    grads = zeros(state["blocks"])
    settings = state["model"].get("contrastive")
    seeds = [[[0.0] * state["model"]["dim"] for _ in f["_active"]] for f in forwards]
    contrastive = None
    if settings is not None:
        ctape = objective.values(forwards, settings)
        contrastive = ctape["loss"]
        for (mi, ti), g in zip(ctape["locations"], objective.pullback(ctape)):
            seeds[mi][ti] = [v * W * float(settings["coefficient"]) for v in g]
    local_statuses = []
    for mi, fwd in enumerate(forwards):
        raw = network.backward(state, cfg, fwd)
        local_statuses.append(amp_microbatch_status(trainable_values(state["blocks"], raw),
                                                   state["loss_scaler"]["scale"]))
        if W:
            local = network.backward(state, cfg, fwd, loads, seeds[mi])
            for dst, source in zip(grads, local):
                for i, value in enumerate(source):
                    dst[i] += value / W
    task = math.fsum(f["token_weight"] * f["task_loss"] for f in forwards) / W if W else 0.0
    total = task + math.fsum(float(layer["aux_coef"]) * report["aux_loss"]
                            + float(layer["z_coef"]) * report["z_loss"]
                            for layer, report in zip(state["model"]["layers"], layers))
    stats = {"microbatches": [public_forward(f) for f in forwards], "layers": layers,
             "token_weight": W, "task_loss": task, "total_loss": total}
    if contrastive is not None:
        stats["contrastive_loss"] = contrastive
        stats["total_loss"] += float(settings["coefficient"]) * contrastive
    return stats, grads, local_statuses
