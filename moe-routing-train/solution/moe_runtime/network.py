"""Residual MoE execution and parameter/input pullbacks.

The model stores D-wide residuals. A layer has an RMSNorm gain view, an
E-by-(D+1) router view and E logical experts. Each expert has an H-by-D
up view, H bias view and D-by-H down view; views use element strides.
Optional sources index the input node (0) or an earlier layer output
(li+1); a mix view supplies trainable join logits. The readout is a fixed
D-vector, not a parameter block. Rank metadata
controls placement only. This module performs no optimizer updates.
"""
import copy
import math
from .ops import softmax_with_lse
from .rng import layer_router_jitter, layer_expert_keep
from .storage import gather, scatter_add, zeros


def _sum(xs):
    return math.fsum(xs)


def _mv(matrix, rows, cols, vector):
    return [_sum(matrix[r * cols + c] * vector[c] for c in range(cols))
            for r in range(rows)]


def _mtv(matrix, rows, cols, vector):
    return [_sum(matrix[r * cols + c] * vector[r] for r in range(rows))
            for c in range(cols)]


def _outer(a, b):
    return [x * y for x in a for y in b]


def forward(state, cfg, window_id, mb):
    model = state["model"]
    blocks = state["blocks"]
    D = int(model["dim"])
    active = sorted((copy.deepcopy(t) for t in mb["tokens"] if t["mask"]),
                    key=lambda t: int(t["ordinal"]))
    weight = _sum(float(t["weight"]) for t in active)
    vectors = [[float(x) for x in t["x"]] for t in active]
    nodes = [vectors]
    tapes = []
    reports = []
    for li, layer in enumerate(model["layers"]):
        source_ids = layer.get("sources", [li])
        mix = softmax_with_lse(gather(blocks, layer["mix"]))[0] if "mix" in layer else [1.0]
        vectors = [[_sum(g * nodes[src][ti][d] for src, g in zip(source_ids, mix))
                    for d in range(D)] for ti in range(len(active))]
        experts = layer["experts"]
        E = len(experts)
        k = int(layer["top_k"])
        gain = gather(blocks, layer["norm"])
        router = gather(blocks, layer["router"])
        capacity = (max(1, math.ceil(float(layer["capacity_factor"]) * len(active) * k / E))
                    if active else 0)
        queues = [[] for _ in experts]
        rows = []
        for ri, (token, x) in enumerate(zip(active, vectors)):
            inv_rms = 1.0 / math.sqrt(_sum(v * v for v in x) / D + float(layer["rms_epsilon"]))
            u = [x[d] * inv_rms * gain[d] for d in range(D)]
            logits = [(_sum(router[e * (D + 1) + d] * u[d] for d in range(D))
                       + router[e * (D + 1) + D]
                       + layer_router_jitter(cfg["rng_seed"], layer["router_jitter"],
                                             window_id, mb["id"], token["uid"], layer["id"], e))
                      for e in range(E)]
            probs, lse = softmax_with_lse(logits)
            selected = sorted(range(E), key=lambda e: logits[e], reverse=True)[:k]
            mass = _sum(probs[e] for e in selected)
            for slot, e in enumerate(selected):
                queues[e].append((-probs[e] / mass, int(token["ordinal"]), ri, slot))
            rows.append({"x": x, "u": u, "inv_rms": inv_rms, "p": probs,
                         "lse": lse, "selected": selected, "admitted": [False] * k})
        accepted_counts = [0] * E
        dropped_counts = [0] * E
        for e, queue in enumerate(queues):
            queue.sort()
            for pos, (_, _, ri, slot) in enumerate(queue):
                if pos < capacity:
                    rows[ri]["admitted"][slot] = True
                    accepted_counts[e] += 1
                else:
                    dropped_counts[e] += 1
        routes = []
        prob_sum = [0.0] * E
        accepted_weight = [0.0] * E
        vectors = []
        for token, row in zip(active, rows):
            wt = float(token["weight"])
            for e in range(E):
                prob_sum[e] += wt * row["p"][e]
            accepted = [e for slot, e in enumerate(row["selected"]) if row["admitted"][slot]]
            denom = _sum(row["p"][e] for e in accepted) if accepted else 1.0
            gates = [row["p"][e] / denom for e in accepted]
            expert_tapes = []
            for e in accepted:
                accepted_weight[e] += wt
                expert = experts[e]
                H = int(expert["up"]["shape"][0])
                up = gather(blocks, expert["up"])
                bias = gather(blocks, expert["bias"])
                down = gather(blocks, expert["down"])
                pre = _mv(up, H, D, row["u"])
                hidden = [math.tanh(pre[h] + bias[h]) for h in range(H)]
                dropout = float(layer["expert_dropout"])
                mask = [layer_expert_keep(cfg["rng_seed"], dropout, window_id, mb["id"],
                                          token["uid"], layer["id"], e, h) for h in range(H)]
                factors = [(1.0 / (1.0 - dropout) if keep else 0.0) for keep in mask]
                dropped = [z * factor for z, factor in zip(hidden, factors)]
                y = _mv(down, D, H, dropped)
                expert_tapes.append({"expert": e, "hidden": hidden, "factors": factors,
                                     "dropped": dropped, "y": y})
            mixture = [_sum(g * tape["y"][d] for g, tape in zip(gates, expert_tapes))
                       for d in range(D)]
            vectors.append([row["x"][d] + float(layer["residual_scale"]) * mixture[d]
                            for d in range(D)])
            row.update(accepted=accepted, gates=gates, experts=expert_tapes, mixture=mixture)
            routes.append({"ordinal": int(token["ordinal"]), "uid": str(token["uid"]),
                           "selected_experts": row["selected"], "accepted_experts": accepted,
                           "accepted_gates": gates})
        local_z = (_sum(float(t["weight"]) * r["lse"] ** 2 for t, r in zip(active, rows)) / weight
                   if weight else 0.0)
        reports.append({"layer_id": layer["id"], "capacity": capacity,
                        "accepted_counts": accepted_counts, "dropped_counts": dropped_counts,
                        "z_loss": local_z, "routes": routes})
        tapes.append({"rows": rows, "prob_sum": prob_sum, "accepted_weight": accepted_weight,
                      "sources": source_ids, "mix": mix})
        nodes.append(vectors)
    predictions = [_sum(x * y for x, y in zip(vector, model["readout"])) for vector in vectors]
    errors = [pred - float(t["target"]) for pred, t in zip(predictions, active)]
    task_sum = _sum(float(t["weight"]) * 0.5 * err * err for t, err in zip(active, errors))
    return {"microbatch_id": mb["id"], "active_tokens": len(active), "token_weight": weight,
            "task_loss": task_sum / weight if weight else 0.0,
            "layers": reports,
            "predictions": [{"ordinal": int(t["ordinal"]), "uid": str(t["uid"]), "value": p}
                            for t, p in zip(active, predictions)],
            "_active": active, "_errors": errors, "_tapes": tapes, "_vectors": vectors, "_nodes": nodes}


def window_statistics(model, forwards):
    """Detached admission statistics for the entire optimizer window."""
    W = _sum(f["token_weight"] for f in forwards)
    loads = []
    reports = []
    for li, layer in enumerate(model["layers"]):
        E = len(layer["experts"])
        k = int(layer["top_k"])
        pbar = [(_sum(f["_tapes"][li]["prob_sum"][e] for f in forwards) / W if W else 0.0)
                for e in range(E)]
        load = [(_sum(f["_tapes"][li]["accepted_weight"][e] for f in forwards) / (W * k)
                 if W else 0.0) for e in range(E)]
        aux = E * _sum(p * a for p, a in zip(pbar, load))
        z = (_sum(f["token_weight"] * f["layers"][li]["z_loss"] for f in forwards) / W
             if W else 0.0)
        loads.append(load)
        reports.append({"layer_id": layer["id"], "aux_loss": aux, "z_loss": z,
                        "probability_mean": pbar, "accepted_load": load})
    return W, loads, reports


def backward(state, cfg, fwd, loads=None, output_adjoints=None):
    """Raw task + per-layer z pullback, plus auxiliary pullbacks when supplied."""
    model = state["model"]
    blocks = state["blocks"]
    D = int(model["dim"])
    gradients = zeros(blocks)
    for ti, token in enumerate(fwd["_active"]):
        wt = float(token["weight"])
        dx = [wt * fwd["_errors"][ti] * float(v) for v in model["readout"]]
        if output_adjoints is not None:
            dx = [a + b for a, b in zip(dx, output_adjoints[ti])]
        node_adjoints = [[0.0] * D for _ in fwd["_nodes"]]
        node_adjoints[-1] = dx
        for li in reversed(range(len(model["layers"]))):
            dx = node_adjoints[li + 1]
            layer = model["layers"][li]
            row = fwd["_tapes"][li]["rows"][ti]
            experts = layer["experts"]
            E = len(experts)
            router = gather(blocks, layer["router"])
            gain = gather(blocks, layer["norm"])
            dmixture = [float(layer["residual_scale"]) * v for v in dx]
            du = [0.0] * D
            dl = [0.0] * E
            for gate, tape in zip(row["gates"], row["experts"]):
                e = tape["expert"]
                expert = experts[e]
                H = len(tape["hidden"])
                up = gather(blocks, expert["up"])
                down = gather(blocks, expert["down"])
                dl[e] += gate * _sum(dmixture[d] * (tape["y"][d] - row["mixture"][d])
                                    for d in range(D))
                dy = [gate * v for v in dmixture]
                scatter_add(gradients, expert["down"], _outer(dy, tape["dropped"]))
                dh = _mtv(down, D, H, dy)
                da = [dh[h] * tape["factors"][h] * (1.0 - tape["hidden"][h] ** 2)
                      for h in range(H)]
                scatter_add(gradients, expert["bias"], da)
                scatter_add(gradients, expert["up"], _outer(da, row["u"]))
                expert_du = _mtv(up, H, D, da)
                du = [a + b for a, b in zip(du, expert_du)]
            load = loads[li] if loads is not None else [0.0] * E
            expected_load = _sum(p * a for p, a in zip(row["p"], load))
            for e in range(E):
                dl[e] += (2.0 * float(layer["z_coef"]) * wt * row["lse"] * row["p"][e]
                          + float(layer["aux_coef"]) * E * wt * row["p"][e]
                          * (load[e] - expected_load))
            scatter_add(gradients, layer["router"], _outer(dl, row["u"] + [1.0]))
            for d in range(D):
                du[d] += _sum(router[e * (D + 1) + d] * dl[e] for e in range(E))
            inv = row["inv_rms"]
            scatter_add(gradients, layer["norm"], [du[d] * row["x"][d] * inv for d in range(D)])
            radial = _sum(du[d] * gain[d] * row["x"][d] for d in range(D))
            dx = [dx[d] + du[d] * gain[d] * inv - row["x"][d] * inv ** 3 * radial / D
                  for d in range(D)]
            sources = fwd["_tapes"][li]["sources"]
            mix = fwd["_tapes"][li]["mix"]
            if "mix" in layer:
                dmix = [g * _sum(dx[d] * (fwd["_nodes"][src][ti][d] - row["x"][d]) for d in range(D))
                        for src, g in zip(sources, mix)]
                scatter_add(gradients, layer["mix"], dmix)
            for src, gate in zip(sources, mix):
                for d in range(D):
                    node_adjoints[src][d] += gate * dx[d]
    return gradients
