"""Portable optimizer-window transactions, including sharpness probes."""
import copy
import math
from . import network, window, sharpness
from .ops import amp_microbatch_status, amp_window_status
from .optimizer import adamw, ema
from .storage import zeros, trainable_values
from .window import public_forward


class Session:
    def __init__(self, workload):
        self.workload = copy.deepcopy(workload)
        self.cfg = self.workload["config"]
        self.state = {k: copy.deepcopy(self.workload[k])
                      for k in ("blocks", "model", "scheduler", "loss_scaler")}
        self.windows = {str(w["id"]): w for w in self.workload["windows"]}
        self.pending, self.archive, self.committed, self.prepared = {}, {}, {}, {}
        self.probes, self.probe_archive = {}, {}
        self.completed = []

    def _microbatch(self, wid, mid):
        return next(m for m in self.windows[wid]["microbatches"] if str(m["id"]) == mid)

    def _check_window(self, wid):
        ids = [str(w["id"]) for w in self.workload["windows"]]
        index = ids.index(wid)
        if any(other not in self.committed for other in ids[:index]):
            raise ValueError("the previous optimizer window has not committed")

    def _items(self, wid):
        self._check_window(wid)
        items = [self.pending.get(wid, {}).get(str(m["id"])) for m in self.windows[wid]["microbatches"]]
        if any(item is None or not item["backward"] for item in items):
            raise ValueError("window is missing backward events")
        return items

    def forward(self, window_id, microbatch_id):
        wid, mid = str(window_id), str(microbatch_id)
        if mid in self.archive.get(wid, {}):
            return copy.deepcopy(self.archive[wid][mid])
        self._check_window(wid)
        bucket = self.pending.setdefault(wid, {})
        if mid not in bucket:
            fwd = network.forward(self.state, self.cfg, wid, self._microbatch(wid, mid))
            bucket[mid] = {"forward": fwd, "backward": False}
        return public_forward(bucket[mid]["forward"])

    def backward(self, window_id, microbatch_id):
        wid, mid = str(window_id), str(microbatch_id)
        if wid in self.committed:
            if mid not in self.archive.get(wid, {}):
                raise ValueError("unknown microbatch")
            return None
        item = self.pending[wid][mid]
        if not item["backward"]:
            gradients = network.backward(self.state, self.cfg, item["forward"])
            peak, overflow = amp_microbatch_status(
                trainable_values(self.state["blocks"], gradients), self.state["loss_scaler"]["scale"])
            item.update(backward=True, local_peak=peak, local_overflow=overflow)
        return None

    def ascend(self, window_id):
        wid = str(window_id)
        if wid in self.probe_archive:
            return copy.deepcopy(self.probe_archive[wid])
        if wid in self.probes:
            return copy.deepcopy(self.probes[wid]["receipt"])
        settings = self.cfg["sharpness"]
        items = self._items(wid)
        stats, grads, _ = window.evaluate(self.state, self.cfg, self.windows[wid],
                                          [item["forward"] for item in items])
        peak, overflow = amp_window_status([(i["local_peak"], i["local_overflow"]) for i in items],
                                           trainable_values(self.state["blocks"], grads),
                                           self.state["loss_scaler"]["scale"])
        delta = zeros(self.state["blocks"]) if overflow else sharpness.displacement(self.state["blocks"], grads, settings)
        report = sharpness.receipt(self.windows[wid]["id"], stats, grads, self.state["blocks"], delta, peak, overflow)
        self.probes[wid] = {"receipt": report, "base": stats, "base_gradients": grads,
                            "state": sharpness.perturbed(self.state, delta)}
        return copy.deepcopy(report)

    def prepare(self, window_id):
        wid = str(window_id)
        if wid in self.committed:
            return copy.deepcopy(self.committed[wid])
        if wid in self.prepared:
            return copy.deepcopy(self.prepared[wid]["report"])
        items = self._items(wid)
        base_forwards = [item["forward"] for item in items]
        extra = None
        if "sharpness" in self.cfg:
            self.ascend(wid)
            probe = self.probes[wid]
            ascent = probe["receipt"]
            if ascent["base_overflow"]:
                stats, grads = probe["base"], probe["base_gradients"]
                peak, overflow = ascent["base_scaled_grad_peak"], True
                descent_evaluated = False
            else:
                stats, grads, local = window.evaluate(probe["state"], self.cfg, self.windows[wid], base_forwards)
                peak, overflow = amp_window_status(local, trainable_values(self.state["blocks"], grads),
                                                   self.state["loss_scaler"]["scale"])
                peak = max(peak, ascent["base_scaled_grad_peak"])
                descent_evaluated = True
            extra = {"ascent": copy.deepcopy(ascent), "base": copy.deepcopy(probe["base"]),
                     "descent_evaluated": descent_evaluated}
        else:
            stats, grads, _ = window.evaluate(self.state, self.cfg, self.windows[wid], base_forwards)
            peak, overflow = amp_window_status([(i["local_peak"], i["local_overflow"]) for i in items],
                                               trainable_values(self.state["blocks"], grads),
                                               self.state["loss_scaler"]["scale"])
        values = trainable_values(self.state["blocks"], grads)
        state_after = copy.deepcopy(self.probes[wid]["state"] if wid in self.probes else self.state)
        scaler, scheduler = state_after["loss_scaler"], state_after["scheduler"]
        scale, cursor = float(scaler["scale"]), int(scheduler["cursor"])
        norm = math.hypot(*values)
        clip = None if overflow else min(1.0, float(self.cfg["max_grad_norm"]) / (norm + 1e-30))
        lr = float(self.cfg["optimizer"]["base_lr"]) * float(scheduler["multipliers"][cursor])
        if overflow:
            scaler["scale"] = max(float(scaler["min_scale"]), scale * float(scaler["backoff_factor"]))
            scaler["growth_tracker"] = 0
        else:
            for block, grad in zip(state_after["blocks"], grads):
                if block.get("trainable", True):
                    adamw(block, [v * clip for v in grad], self.cfg["optimizer"], lr)
                    ema(block, self.cfg["ema_decay"])
            scheduler["cursor"] = (cursor + 1) % len(scheduler["multipliers"])
            scaler["growth_tracker"] += 1
            if scaler["growth_tracker"] >= int(scaler["growth_interval"]):
                scaler["scale"] = min(float(scaler["max_scale"]), scale * float(scaler["growth_factor"]))
                scaler["growth_tracker"] = 0
        report = {"window_id": self.windows[wid]["id"],
                  "active_tokens": sum(f["active_tokens"] for f in stats["microbatches"]),
                  **copy.deepcopy(stats), "aux_loss": math.fsum(r["aux_loss"] for r in stats["layers"]),
                  "z_loss": math.fsum(r["z_loss"] for r in stats["layers"]),
                  "grad_norm": norm, "scaled_grad_peak": peak, "clip_coef": clip,
                  "overflow": overflow, "update_applied": not overflow, "lr": lr,
                  "scheduler_cursor_before": cursor, "scheduler_cursor_after": scheduler["cursor"],
                  "loss_scale_before": scale, "loss_scale_after": scaler["scale"]}
        if extra is not None:
            report["sharpness"] = extra
        self.prepared[wid] = {"report": report, "state": state_after}
        return copy.deepcopy(report)

    def flush(self, window_id):
        wid = str(window_id)
        if wid in self.committed:
            return copy.deepcopy(self.committed[wid])
        self.prepare(wid)
        update = self.prepared.pop(wid)
        self.state = update["state"]
        report = update["report"]
        self.archive[wid] = {mid: public_forward(item["forward"])
                             for mid, item in self.pending.pop(wid, {}).items()}
        if wid in self.probes:
            self.probe_archive[wid] = copy.deepcopy(self.probes.pop(wid)["receipt"])
        self.completed.append(copy.deepcopy(report))
        self.committed[wid] = copy.deepcopy(report)
        return copy.deepcopy(report)

    def snapshot(self):
        return copy.deepcopy(self.state)

    def checkpoint(self):
        return copy.deepcopy({"format": "residual-moe-checkpoint/2", "state": self.state,
                              "pending": self.pending, "archive": self.archive,
                              "completed": self.completed, "committed": self.committed,
                              "prepared": self.prepared, "probes": self.probes,
                              "probe_archive": self.probe_archive})

    @classmethod
    def restore(cls, workload, image):
        if image.get("format") not in ("residual-moe-checkpoint/1", "residual-moe-checkpoint/2"):
            raise ValueError("unsupported checkpoint")
        result = cls(workload)
        for name in ("state", "pending", "archive", "completed", "committed", "prepared"):
            setattr(result, name, copy.deepcopy(image[name]))
        for name in ("probes", "probe_archive"):
            setattr(result, name, copy.deepcopy(image.get(name, {})))
        result.state["blocks"] = copy.deepcopy(workload["blocks"])
        for bucket in result.pending.values():
            for item in bucket.values():
                item["local_peak"] = 0.0
                item["local_overflow"] = False
        result.state["model"] = copy.deepcopy(workload["model"])
        for staged in list(result.prepared.values()) + list(result.probes.values()):
            staged["state"]["model"] = copy.deepcopy(workload["model"])
        return result
