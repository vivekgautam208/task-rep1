"""Fixtures for globally coupled objectives and restartable sharpness training."""
import copy
import random
from stack_cases import make_stack_case


def make_coupled_case(seed=7001, *, adaptive=True, contrastive=True, sharpness=True,
                      D=3, H=4, E=5, depth=3, windows=3, radius=0.22,
                      coefficient=0.85, temperature=0.23, graph=True, **kwargs):
    w = make_stack_case(seed, D=D, H=H, E=E, depth=depth, windows=windows, **kwargs)
    if graph:
        rng = random.Random(seed + 157)
        joins = {}
        for li, layer in enumerate(w["model"]["layers"]):
            if li == 0:
                continue
            sources = [li, 0] if li == 1 else ([li, li // 2] if li % 2 else [li, li - 1, 0])
            layer["sources"] = sources
            n = len(sources)
            if n in joins and li % 3:
                layer["mix"] = copy.deepcopy(joins[n])
            else:
                weights = [rng.uniform(-0.9, 0.9) for _ in range(n + 2)]
                w["blocks"].append({"weight": weights, "m": [0.025] * len(weights),
                                    "v": [0.03] * len(weights), "ema": list(weights),
                                    "step": 3, "lr_scale": 0.8, "decay_scale": 0.6, "trainable": True})
                layer["mix"] = {"block": len(w["blocks"]) - 1, "offset": 1,
                                "shape": [n], "strides": [1]}
                joins[n] = copy.deepcopy(layer["mix"])
    if contrastive:
        w["model"]["contrastive"] = {"coefficient": coefficient, "temperature": temperature,
                                      "normalize_epsilon": 0.025}
    if sharpness:
        w["config"]["sharpness"] = {"radius": radius, "adaptive": adaptive,
                                    "eta": 0.03, "epsilon": 1e-12}
    for wi, window in enumerate(w["windows"]):
        for mi, mb in enumerate(window["microbatches"]):
            for token in mb["tokens"]:
                token["class_id"] = f"class-{(token['ordinal'] + mi) % 4}"
        if sharpness:
            at = next(i for i, event in enumerate(window["events"]) if event["op"] == "prepare")
            window["events"][at:at] = [{"op": "ascend"}, {"op": "ascend"}, {"op": "restart"}]
            window["events"].extend([{"op": "ascend"}])
    return w


def rearrange(workload, seed=71):
    result = copy.deepcopy(workload)
    rng = random.Random(seed)
    for layer in result["model"]["layers"]:
        for expert in layer["experts"]:
            expert["home_rank"] = rng.randrange(17)
    for window in result["windows"]:
        order = window["microbatches"][:]
        rng.shuffle(order)
        for mb in window["microbatches"]:
            rng.shuffle(mb["tokens"])
            for token in mb["tokens"]:
                token["rank"] = rng.randrange(23)
        events = [{"op": "forward", "microbatch_id": m["id"]} for m in order]
        events.append({"op": "restart"})
        rng.shuffle(order)
        events += [{"op": "backward", "microbatch_id": m["id"]} for m in order]
        if "sharpness" in result["config"]:
            events += [{"op": "ascend"}, {"op": "restart"}]
        events += [{"op": "prepare"}, {"op": "restart"}, {"op": "flush"}, {"op": "flush"}]
        window["events"] = events
    return result
