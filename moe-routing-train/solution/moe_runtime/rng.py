import hashlib

ROUTER_JITTER_LABEL = "router_jitter"
EXPERT_DROPOUT_LABEL = "expert_dropout"

def uniform01(seed, *parts):
    key = "|".join([str(int(seed))] + [str(p) for p in parts]).encode("utf-8")
    raw = hashlib.blake2b(key, digest_size=8, person=b"moe-rng-v1").digest()
    return int.from_bytes(raw, "big") / 18446744073709551616.0

def symmetric(seed, scale, *parts):
    return float(scale) * (2.0 * uniform01(seed, *parts) - 1.0)

def keep(seed, probability, *parts):
    p = float(probability)
    if p <= 0.0:
        return True
    if p >= 1.0:
        return False
    return uniform01(seed, *parts) >= p

def router_jitter(seed, scale, window_id, microbatch_id, uid, expert):
    return symmetric(seed, scale, ROUTER_JITTER_LABEL, str(window_id), str(microbatch_id), str(uid), int(expert))

def expert_keep(seed, probability, window_id, microbatch_id, uid, expert, hidden_index):
    return keep(seed, probability, EXPERT_DROPOUT_LABEL, str(window_id), str(microbatch_id), str(uid), int(expert), int(hidden_index))

class CursorRng:
    """Legacy execution-order RNG retained for compatibility with old callers."""
    def __init__(self, seed):
        self.seed = int(seed)
        self.cursor = 0

    def uniform(self):
        value = uniform01(self.seed, "cursor", self.cursor)
        self.cursor += 1
        return value


def layer_router_jitter(seed, scale, window_id, microbatch_id, uid, layer_id, expert):
    """Layer identity is logical, not a pipeline stage or worker identity."""
    return symmetric(seed, scale, "layer_router_jitter", str(window_id),
                     str(microbatch_id), str(uid), str(layer_id), int(expert))


def layer_expert_keep(seed, probability, window_id, microbatch_id, uid,
                      layer_id, expert, hidden_index):
    return keep(seed, probability, "layer_expert_dropout", str(window_id),
                str(microbatch_id), str(uid), str(layer_id), int(expert),
                int(hidden_index))
