"""Independent scalar autodiff reference; does not import candidate or oracle code."""
import copy
import hashlib
import math
from reference_model import adam, ema


class Tape:
    def __init__(self):
        self.nodes = []

    def node(self, value, edges=()):
        result = Scalar(self, float(value), edges)
        self.nodes.append(result)
        return result

    def total(self, values):
        terms = list(values)
        return self.node(math.fsum(t.value if isinstance(t, Scalar) else t for t in terms),
                         [(t, 1.0) for t in terms if isinstance(t, Scalar)])

    def gradient(self, output, parameters):
        for node in self.nodes:
            node.adjoint = 0.0
        output.adjoint = 1.0
        for node in reversed(self.nodes):
            for parent, derivative in node.edges:
                parent.adjoint += node.adjoint * derivative
        return [[x.adjoint for x in block] for block in parameters]


class Scalar:
    def __init__(self, tape, value, edges):
        self.tape, self.value, self.edges = tape, value, edges
        self.adjoint = 0.0

    def __add__(self, other):
        if isinstance(other, Scalar):
            return self.tape.node(self.value + other.value, [(self, 1.0), (other, 1.0)])
        return self.tape.node(self.value + other, [(self, 1.0)])

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return self.tape.node(self.value * other.value, [(self, other.value), (other, self.value)])
        return self.tape.node(self.value * other, [(self, other)])

    __rmul__ = __mul__

    def __pow__(self, power):
        return self.tape.node(self.value ** power, [(self, power * self.value ** (power - 1))])

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return self ** -1 * other

    def exp(self):
        value = math.exp(self.value)
        return self.tape.node(value, [(self, value)])

    def log(self):
        return self.tape.node(math.log(self.value), [(self, 1 / self.value)])

    def tanh(self):
        value = math.tanh(self.value)
        return self.tape.node(value, [(self, 1 - value * value)])


def uniform(seed, *coordinates):
    raw = "|".join(str(x) for x in (int(seed),) + coordinates).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8, person=b"moe-rng-v1").digest()
    return int.from_bytes(digest, "big") / 2**64


def logical(parameters, view):
    indices = [view.get("offset", 0)]
    for length, stride in zip(view["shape"], view["strides"]):
        indices = [base + n * stride for base in indices for n in range(length)]
    return [parameters[view["block"]][i] for i in indices]


def evaluate_window(state, config, window):
    tape = Tape()
    params = [[tape.node(v) for v in b["weight"]] for b in state["blocks"]]
    model = state["model"]
    dim = model["dim"]
    micros = []
    probabilities = [[] for _ in model["layers"]]
    accepted_weights = [[0.0] * len(l["experts"]) for l in model["layers"]]
    z_sums = [[] for _ in model["layers"]]
    raw_losses = []
    raw_tasks = []
    weights = []
    bank = []
    for mb in window["microbatches"]:
        tokens = sorted([t for t in mb["tokens"] if t["mask"]], key=lambda t: int(t["ordinal"]))
        W = math.fsum(t["weight"] for t in tokens)
        weights.append(W)
        vectors = [[tape.node(x) for x in t["x"]] for t in tokens]
        node_vectors = [vectors]
        layer_reports = []
        local_z_terms = []
        for li, layer in enumerate(model["layers"]):
            sources = layer.get("sources", [li])
            if "mix" in layer:
                mixing = logical(params, layer["mix"])
                pivot = max(a.value for a in mixing)
                exp_mixing = [(a - pivot).exp() for a in mixing]
                denominator = tape.total(exp_mixing)
                mixing = [a / denominator for a in exp_mixing]
            else:
                mixing = [tape.node(1.0)]
            vectors = [[tape.total(g * node_vectors[src][ti][d] for src, g in zip(sources, mixing))
                        for d in range(dim)] for ti in range(len(tokens))]
            E = len(layer["experts"])
            K = layer["top_k"]
            gamma = logical(params, layer["norm"])
            R = logical(params, layer["router"])
            cap = max(1, math.ceil(layer["capacity_factor"] * len(tokens) * K / E)) if tokens else 0
            rows = []
            requests = [[] for _ in range(E)]
            for ri, (token, vector) in enumerate(zip(tokens, vectors)):
                inv = (tape.total(x * x for x in vector) / dim + layer["rms_epsilon"]) ** -0.5
                normalized = [vector[d] * inv * gamma[d] for d in range(dim)]
                logits = []
                for expert in range(E):
                    noise = layer["router_jitter"] * (2 * uniform(
                        config["rng_seed"], "layer_router_jitter", str(window["id"]), str(mb["id"]),
                        str(token["uid"]), str(layer["id"]), expert) - 1)
                    logits.append(tape.total(R[expert * (dim + 1) + d] * normalized[d]
                                             for d in range(dim)) + R[expert * (dim + 1) + dim] + noise)
                peak = max(x.value for x in logits)
                exponentials = [(x - peak).exp() for x in logits]
                denominator = tape.total(exponentials)
                p = [x / denominator for x in exponentials]
                lse = denominator.log() + peak
                chosen = sorted(range(E), key=lambda e: logits[e].value, reverse=True)[:K]
                mass = math.fsum(p[e].value for e in chosen)
                for slot, e in enumerate(chosen):
                    requests[e].append((-p[e].value / mass, token["ordinal"], ri, slot))
                rows.append({"norm": normalized, "p": p, "lse": lse, "selected": chosen, "slots": set()})
            ac, dr = [0] * E, [0] * E
            for e, requests_e in enumerate(requests):
                for pos, (_, _, ri, slot) in enumerate(sorted(requests_e)):
                    if pos < cap:
                        rows[ri]["slots"].add(slot)
                        ac[e] += 1
                    else:
                        dr[e] += 1
            new_vectors = []
            routes = []
            local_z = []
            for token, vector, row in zip(tokens, vectors, rows):
                wt = token["weight"]
                probabilities[li].append((wt, row["p"]))
                accepted = [e for slot, e in enumerate(row["selected"]) if slot in row["slots"]]
                denominator = tape.total(row["p"][e] for e in accepted) if accepted else tape.node(1)
                gates = [row["p"][e] / denominator for e in accepted]
                outputs = []
                for expert_id in accepted:
                    accepted_weights[li][expert_id] += wt
                    expert = layer["experts"][expert_id]
                    H = expert["up"]["shape"][0]
                    up, bias, down = [logical(params, expert[field]) for field in ("up", "bias", "down")]
                    hidden = [(tape.total(up[h * dim + d] * row["norm"][d] for d in range(dim)) + bias[h]).tanh()
                              for h in range(H)]
                    pdrop = layer["expert_dropout"]
                    dropped = []
                    for h, activation in enumerate(hidden):
                        r = uniform(config["rng_seed"], "layer_expert_dropout", str(window["id"]),
                                    str(mb["id"]), str(token["uid"]), str(layer["id"]), expert_id, h)
                        dropped.append(activation * (1 / (1 - pdrop) if r >= pdrop else 0.0))
                    outputs.append([tape.total(down[d * H + h] * dropped[h] for h in range(H))
                                    for d in range(dim)])
                new_vectors.append([vector[d] + layer["residual_scale"] * tape.total(
                    gate * output[d] for gate, output in zip(gates, outputs)) for d in range(dim)])
                z = row["lse"] * row["lse"] * wt
                local_z.append(z)
                z_sums[li].append(z)
                routes.append({"ordinal": int(token["ordinal"]), "uid": str(token["uid"]),
                               "selected_experts": row["selected"], "accepted_experts": accepted,
                               "accepted_gates": [g.value for g in gates]})
            vectors = new_vectors
            node_vectors.append(vectors)
            ztotal = tape.total(local_z)
            local_z_terms.append(ztotal * layer["z_coef"])
            layer_reports.append({"layer_id": layer["id"], "capacity": cap, "accepted_counts": ac,
                                  "dropped_counts": dr, "z_loss": ztotal.value / W if W else 0.0,
                                  "routes": routes})
        bank.extend((token, vector) for token, vector in zip(tokens, vectors))
        predictions = [tape.total(a * b for a, b in zip(vector, model["readout"])) for vector in vectors]
        task = tape.total((prediction - token["target"]) ** 2 * (0.5 * token["weight"])
                          for prediction, token in zip(predictions, tokens))
        raw_tasks.append(task)
        raw_losses.append(task + tape.total(local_z_terms))
        micros.append({"microbatch_id": mb["id"], "active_tokens": len(tokens), "token_weight": W,
                       "task_loss": task.value / W if W else 0.0, "layers": layer_reports,
                       "predictions": [{"ordinal": int(token["ordinal"]), "uid": str(token["uid"]), "value": p.value}
                                       for token, p in zip(tokens, predictions)]})
    W = math.fsum(weights)
    task = tape.total(raw_tasks) / W if W else tape.node(0)
    total = task
    layers = []
    for li, layer in enumerate(model["layers"]):
        E, K = len(layer["experts"]), layer["top_k"]
        pbar = [(tape.total(wt * ps[e] for wt, ps in probabilities[li]) / W if W else tape.node(0))
                for e in range(E)]
        loads = [x / (W * K) if W else 0.0 for x in accepted_weights[li]]
        aux = tape.total(p * load for p, load in zip(pbar, loads)) * E
        z = tape.total(z_sums[li]) / W if W else tape.node(0)
        total = total + aux * layer["aux_coef"] + z * layer["z_coef"]
        layers.append({"layer_id": layer["id"], "aux_loss": aux.value, "z_loss": z.value,
                       "probability_mean": [p.value for p in pbar], "accepted_load": loads})
    contrastive_value = None
    if "contrastive" in model:
        head = model["contrastive"]
        normalized = []
        for token, vector in bank:
            inv = (tape.total(x * x for x in vector) + head["normalize_epsilon"]) ** -0.5
            normalized.append([x * inv for x in vector])
        terms, masses = [], []
        for i, (token, _) in enumerate(bank):
            positive_ids = [j for j, (other, _) in enumerate(bank)
                            if j != i and other["class_id"] == token["class_id"]]
            if not positive_ids:
                continue
            all_ids = [j for j in range(len(bank)) if j != i]
            scores = {j: tape.total(a * b for a, b in zip(normalized[i], normalized[j])) / head["temperature"]
                      for j in all_ids}
            pivot = max(score.value for score in scores.values())
            lse = tape.total((score - pivot).exp() for score in scores.values()).log() + pivot
            loss_i = lse - tape.total(scores[j] for j in positive_ids) / len(positive_ids)
            terms.append(loss_i * token["weight"])
            masses.append(token["weight"])
        mass = math.fsum(masses)
        contrastive = tape.total(terms) / mass if mass else tape.node(0)
        contrastive_value = contrastive.value
        total = total + contrastive * head["coefficient"]
    local_gradients = [tape.gradient(loss, params) for loss in raw_losses]
    gradients = tape.gradient(total, params)
    stats = {"microbatches": micros, "layers": layers, "token_weight": W,
             "task_loss": task.value, "total_loss": total.value}
    if contrastive_value is not None:
        stats["contrastive_loss"] = contrastive_value
    return stats, gradients, local_gradients



def gradient_status(state, gradients, local):
    selected = [i for i, b in enumerate(state["blocks"]) if b.get("trainable", True)]
    scale = float(state["loss_scaler"]["scale"])
    scaled = [abs(g * scale) for groups in [gradients] + local for i in selected for g in groups[i]]
    return max(scaled, default=0.0), any(not math.isfinite(x) or x > 65504 for x in scaled)


def probe_state(state, config, window):
    stats, gradients, local = evaluate_window(state, config, window)
    peak, overflow = gradient_status(state, gradients, local)
    selected = [i for i, b in enumerate(state["blocks"]) if b.get("trainable", True)]
    normal = math.hypot(*(g for i in selected for g in gradients[i]))
    settings = config["sharpness"]
    result = copy.deepcopy(state)
    changes = [[0.0] * len(b["weight"]) for b in state["blocks"]]
    if not overflow:
        metric = {(bi, j): (abs(w) + settings["eta"] if settings["adaptive"] else 1.0)
                  for bi in selected for j, w in enumerate(state["blocks"][bi]["weight"])}
        denominator = math.sqrt(math.fsum((metric[bi, j] * gradients[bi][j]) ** 2
                                          for bi, j in metric)) + settings["epsilon"]
        for bi, j in metric:
            changes[bi][j] = settings["radius"] * metric[bi, j] ** 2 * gradients[bi][j] / denominator
            result["blocks"][bi]["weight"][j] += changes[bi][j]
    receipt = {"window_id": window["id"], "base_total_loss": stats["total_loss"],
               "base_grad_norm": normal, "perturbation_norm": math.hypot(*(d for ds in changes for d in ds)),
               "base_scaled_grad_peak": peak, "base_overflow": overflow}
    return result, stats, gradients, receipt


def reference_stack(workload):
    state = {k: copy.deepcopy(workload[k]) for k in ("blocks", "model", "scheduler", "loss_scaler")}
    cfg = workload["config"]
    steps = []
    for window in workload["windows"]:
        extra = None
        if "sharpness" in cfg:
            perturbed, base, base_grad, ascent = probe_state(state, cfg, window)
            if ascent["base_overflow"]:
                stats, grads = base, base_grad
                peak, overflow, evaluated = ascent["base_scaled_grad_peak"], True, False
            else:
                stats, grads, local = evaluate_window(perturbed, cfg, window)
                peak, overflow = gradient_status(perturbed, grads, local)
                peak = max(peak, ascent["base_scaled_grad_peak"])
                evaluated = True
            extra = {"ascent": ascent, "base": base, "descent_evaluated": evaluated}
        else:
            stats, grads, local = evaluate_window(state, cfg, window)
            peak, overflow = gradient_status(state, grads, local)
        selected = [i for i, b in enumerate(state["blocks"]) if b.get("trainable", True)]
        values = [g for i in selected for g in grads[i]]
        scale = float(state["loss_scaler"]["scale"])
        norm = math.hypot(*values)
        coef = None if overflow else min(1.0, cfg["max_grad_norm"] / (norm + 1e-30))
        scheduler, scaler = state["scheduler"], state["loss_scaler"]
        cursor = scheduler["cursor"]
        lr = cfg["optimizer"]["base_lr"] * scheduler["multipliers"][cursor]
        if overflow:
            scaler["scale"] = max(scaler["min_scale"], scale * scaler["backoff_factor"])
            scaler["growth_tracker"] = 0
        else:
            for i in selected:
                adam(state["blocks"][i], [x * coef for x in grads[i]], cfg["optimizer"], lr)
                ema(state["blocks"][i], cfg["ema_decay"])
            scheduler["cursor"] = (cursor + 1) % len(scheduler["multipliers"])
            scaler["growth_tracker"] += 1
            if scaler["growth_tracker"] >= scaler["growth_interval"]:
                scaler["scale"] = min(scaler["max_scale"], scale * scaler["growth_factor"])
                scaler["growth_tracker"] = 0
        report = {"window_id": window["id"], "active_tokens": sum(m["active_tokens"] for m in stats["microbatches"]),
                  **stats, "aux_loss": math.fsum(l["aux_loss"] for l in stats["layers"]),
                  "z_loss": math.fsum(l["z_loss"] for l in stats["layers"]),
                  "grad_norm": norm, "scaled_grad_peak": peak, "clip_coef": coef,
                  "overflow": overflow, "update_applied": not overflow, "lr": lr,
                  "scheduler_cursor_before": cursor, "scheduler_cursor_after": scheduler["cursor"],
                  "loss_scale_before": scale, "loss_scale_after": scaler["scale"]}
        if extra is not None:
            report["sharpness"] = extra
        steps.append(report)
    return {"steps": steps, "final": state}
