"""Deterministic production-shaped model, storage and event fixtures."""
import copy
import random


def view(block, offset, shape, strides):
    return {"block": block, "offset": offset, "shape": shape, "strides": strides}


def make_stack_case(seed=1, D=3, H=4, E=5, depth=3, windows=3,
                    shared=True, overlap=True, dropout=0.25, pressure=True,
                    frozen=True, restarts=True, norm_eps=0.08):
    rnd = random.Random(seed)
    blocks = []

    def block(n, gain=False):
        weight = [rnd.uniform(0.7, 1.25) if gain else rnd.uniform(-0.65, 0.65) for _ in range(n)]
        blocks.append({"weight": weight, "m": [rnd.uniform(-0.03, 0.03) for _ in range(n)],
                       "v": [rnd.uniform(0.002, 0.05) for _ in range(n)],
                       "ema": [v + rnd.uniform(-0.02, 0.02) for v in weight],
                       "step": rnd.randrange(1, 7), "lr_scale": rnd.choice([0.6, 1.0, 1.4]),
                       "decay_scale": rnd.choice([0.0, 0.5, 1.0]),
                       "trainable": not (frozen and len(blocks) % 7 == 4)})
        return len(blocks) - 1

    layers = []
    expert_pool = []
    for li in range(depth):
        gamma_id = block(D + 2, gain=True)
        router_id = block(E * (D + 1) + 3)
        # Transposed storage is a common fused-parameter layout.
        router_view = view(router_id, 1, [E, D + 1], [1, E])
        if shared and li and li % 2:
            router_view = copy.deepcopy(layers[li - 1]["router"])
        experts = []
        for e in range(E):
            if shared and expert_pool and e % 3 == 2:
                experts.append(copy.deepcopy(expert_pool[(e + li) % len(expert_pool)]))
                continue
            if overlap:
                pid = block(H * D + H + 4)
                up = view(pid, 1, [H, D], [D, 1])
                down = view(pid, 1, [D, H], [1, D])
                bias = view(pid, H * D + 2, [H], [1])
            else:
                up_id, down_id, bias_id = block(H * D), block(D * H), block(H)
                up = view(up_id, 0, [H, D], [D, 1])
                down = view(down_id, 0, [D, H], [H, 1])
                bias = view(bias_id, 0, [H], [1])
            expert = {"up": up, "down": down, "bias": bias, "home_rank": rnd.randrange(4)}
            experts.append(expert)
            expert_pool.append(copy.deepcopy(expert))
        layers.append({"id": f"block-{li}", "norm": view(gamma_id, 1, [D], [1]),
                       "router": router_view, "experts": experts,
                       "top_k": min(E, 2 + li % 2), "capacity_factor": 0.62 if pressure else 3.0,
                       "rms_epsilon": norm_eps, "residual_scale": rnd.choice([0.4, 0.7, 1.0]),
                       "router_jitter": 0.16, "expert_dropout": dropout,
                       "aux_coef": 0.05 + 0.03 * li, "z_coef": 0.011 + 0.007 * li})
    # A padding-only trainable block is legal storage and still receives AdamW.
    block(5)
    blocks[-1]["trainable"] = True
    ws = []
    for wi in range(windows):
        microbatches = []
        for mi in range(2 + (wi + seed) % 2):
            tokens = [{"uid": f"sample-{wi}-{mi}-{ti}", "ordinal": ti, "rank": rnd.randrange(6),
                       "x": [rnd.uniform(-1.2, 1.2) for _ in range(D)],
                       "target": rnd.uniform(-1.0, 1.0), "weight": rnd.choice([0.15, 0.7, 1.0, 3.5]),
                       "mask": rnd.random() > 0.18}
                      for ti in range(4 + (wi + mi) % 3)]
            rnd.shuffle(tokens)
            microbatches.append({"id": f"micro-{mi}", "tokens": tokens})
        events = []
        order = list(reversed(microbatches))
        for mi, mb in enumerate(order):
            events.append({"op": "forward", "microbatch_id": mb["id"]})
            if restarts and mi == 0:
                events.append({"op": "restart"})
            events.append({"op": "backward", "microbatch_id": mb["id"]})
            events.append({"op": "backward", "microbatch_id": mb["id"]})
        events.extend([{"op": "prepare"}, {"op": "prepare"}])
        if restarts:
            events.append({"op": "restart"})
        events.extend([{"op": "flush"}, {"op": "flush"}])
        if restarts:
            events.append({"op": "restart"})
        events.append({"op": "forward", "microbatch_id": order[0]["id"]})
        events.append({"op": "backward", "microbatch_id": order[0]["id"]})
        ws.append({"id": f"window-{wi}", "microbatches": microbatches, "events": events})
    return {"model": {"dim": D, "layers": layers,
                       "readout": [rnd.uniform(-0.8, 0.8) for _ in range(D)]},
            "blocks": blocks,
            "config": {"rng_seed": seed * 193 + 71, "max_grad_norm": 0.37, "ema_decay": 0.92,
                       "optimizer": {"base_lr": 0.013, "beta1": 0.82, "beta2": 0.95,
                                     "eps": 1e-8, "weight_decay": 0.025}},
            "scheduler": {"cursor": 1, "multipliers": [1.0, 0.7, 1.15, 0.55]},
            "loss_scaler": {"scale": 64.0, "min_scale": 1.0, "max_scale": 131072.0,
                            "growth_tracker": 1, "growth_interval": 2,
                            "growth_factor": 2.0, "backoff_factor": 0.5}, "windows": ws}


def make_stack_cancel_case():
    w = make_stack_case(9917, D=1, H=1, E=1, depth=1, windows=2, shared=False,
                        overlap=False, frozen=False, pressure=False, dropout=0.0)
    def clean(values):
        return {"weight": list(values), "m": [0.0] * len(values), "v": [0.01] * len(values),
                "ema": list(values), "step": 2, "lr_scale": 1.0, "decay_scale": 0.0,
                "trainable": True}
    w["blocks"] = [clean([1.0]), clean([0.0, 0.0]), clean([0.0, 1.0, 0.0])]
    layer = w["model"]["layers"][0]
    layer.update(norm=view(0, 0, [1], [1]), router=view(1, 0, [1, 2], [2, 1]),
                 experts=[{"up": view(2, 0, [1, 1], [1, 1]), "bias": view(2, 1, [1], [1]),
                           "down": view(2, 2, [1, 1], [1, 1]), "home_rank": 0}],
                 residual_scale=1.0, router_jitter=0.0, aux_coef=0.0, z_coef=0.0)
    w["model"]["readout"] = [1.0]
    w["loss_scaler"].update(scale=512.0, growth_tracker=0, growth_interval=10)
    for window in w["windows"]:
        mbs = []
        for mi, target in enumerate([201.0, -199.0]):
            mbs.append({"id": f"cancel-{mi}", "tokens": [
                {"uid": f"cancel-{mi}", "ordinal": 0, "rank": mi, "x": [1.0],
                 "target": target, "weight": 1.0, "mask": True}]})
        window["microbatches"] = mbs
        window["events"] = [event for mb in mbs for event in [
            {"op": "forward", "microbatch_id": mb["id"]},
            {"op": "backward", "microbatch_id": mb["id"]}, {"op": "restart"}]] + [
            {"op": "prepare"}, {"op": "restart"}, {"op": "flush"}]
    return w
