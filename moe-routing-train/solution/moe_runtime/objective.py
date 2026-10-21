"""Window-wide supervised contrastive head used by the training monitor.

Representations belong to active tokens, not ranks. Labels are class IDs;
self identity is the row in the gathered window, not the RNG's token UID.
The scalar value path is shared by monitoring and training. A pullback is
provided for the explicit-vector execution backend.
"""
import math


def values(forwards, config):
    locations = [(mi, ti) for mi, f in enumerate(forwards) for ti in range(len(f["_active"]))]
    vectors = [forwards[mi]["_vectors"][ti] for mi, ti in locations]
    tokens = [forwards[mi]["_active"][ti] for mi, ti in locations]
    eps, temperature = float(config["normalize_epsilon"]), float(config["temperature"])
    norms = [math.sqrt(math.fsum(x * x for x in v) + eps) for v in vectors]
    normalized = [[x / norm for x in v] for v, norm in zip(vectors, norms)]
    rows = []
    for i, token in enumerate(tokens):
        candidates = [j for j in range(len(tokens)) if i != j]
        positives = [j for j in candidates if tokens[j]["class_id"] == token["class_id"]]
        if not positives:
            rows.append(None)
            continue
        logits = [math.fsum(a * b for a, b in zip(normalized[i], normalized[j])) / temperature
                  for j in candidates]
        peak = max(logits)
        exps = [math.exp(s - peak) for s in logits]
        denom = math.fsum(exps)
        probabilities = [x / denom for x in exps]
        positive_mean = math.fsum(s for j, s in zip(candidates, logits) if j in positives) / len(positives)
        rows.append({"candidates": candidates, "positives": positives, "probabilities": probabilities,
                     "loss": peak + math.log(denom) - positive_mean, "weight": float(token["weight"])})
    weight = math.fsum(row["weight"] for row in rows if row is not None)
    loss = math.fsum(row["weight"] * row["loss"] for row in rows if row is not None) / weight if weight else 0.0
    return {"loss": loss, "anchor_weight": weight, "locations": locations, "vectors": vectors,
            "normalized": normalized, "norms": norms, "rows": rows, "temperature": temperature}


def pullback(tape):
    vectors, normalized = tape["vectors"], tape["normalized"]
    grad = [[0.0] * len(v) for v in vectors]
    weight = tape["anchor_weight"]
    if not weight:
        return grad
    for i, row in enumerate(tape["rows"]):
        if row is None:
            continue
        positive_mass = 1.0 / len(row["positives"])
        scale = row["weight"] / (weight * tape["temperature"])
        for j, p in zip(row["candidates"], row["probabilities"]):
            ds = scale * (p - (positive_mass if j in row["positives"] else 0.0))
            for d in range(len(grad[i])):
                grad[i][d] += ds * normalized[j][d]
                grad[j][d] += ds * normalized[i][d]
    answer = []
    for vector, norm, upstream in zip(vectors, tape["norms"], grad):
        radial = math.fsum(x * g for x, g in zip(vector, upstream)) / (norm * norm)
        answer.append([(g - x * radial) / norm for x, g in zip(vector, upstream)])
    return answer
