"""Outcome checks for alias-aware stack training and process-boundary recovery."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import pytest
from stack_cases import make_stack_case
from stack_reference import reference_stack
from test_outputs import close, check, run_cli


def api_chunk(workload, events, previous=None):
    """Drive candidate code outside the verifier identity and import path."""
    request = {"workload": workload, "events": events}
    if previous:
        request.update({k: previous[k] for k in ("image", "receipts", "snapshots")})
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        folder.chmod(0o777)
        inp, out, driver = folder / "request.json", folder / "result.json", folder / "driver.py"
        inp.write_text(json.dumps(request), encoding="utf-8")
        inp.chmod(0o644)
        driver.write_bytes(Path("/tests/stack_probe.py").read_bytes())
        driver.chmod(0o755)
        result = subprocess.run(["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
                                 "--inh-caps=-all", "--no-new-privs", "python3", str(driver), str(inp), str(out)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-4000:]
        return json.loads(out.read_text(encoding="utf-8"))


def journal(workload):
    """Flatten window-local records without changing execution order."""
    return [(w["id"], e) for w in workload["windows"] for e in w["events"]]


def assert_receipts(result, expected):
    """Compare every public forward/prepare/flush result, not only final weights."""
    steps = {str(s["window_id"]): s for s in expected["steps"]}
    for receipt in result["receipts"]:
        step = steps[str(receipt["window_id"])]
        op = receipt["event"]["op"]
        if op == "forward":
            mbid = str(receipt["event"]["microbatch_id"])
            target = next(m for m in step["microbatches"] if str(m["microbatch_id"]) == mbid)
        elif op == "backward":
            target = None
        else:
            target = step
        close(receipt["value"], target)
    close(result["state"], expected["final"])


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_stack_depth_pullbacks(depth):
    """Different depths require complete residual and upstream loss pullbacks."""
    check(make_stack_case(120 + depth, depth=depth, windows=3))


@pytest.mark.parametrize("shape", [(1, 3, 3), (2, 5, 4), (5, 2, 6), (4, 6, 5)])
def test_rectangular_strided_models(shape):
    """Rectangular projections cannot be treated as contiguous square weights."""
    D, H, E = shape
    check(make_stack_case(145 + D, D=D, H=H, E=E, windows=2))


def test_cross_layer_and_logical_aliases():
    """Logical experts and different depths scatter into unique shared storage."""
    check(make_stack_case(201, depth=5, shared=True, overlap=True, windows=5))


def test_independent_unshared_blocks():
    """Correct handling of sharing must not corrupt untied parameter layouts."""
    check(make_stack_case(202, depth=3, shared=False, overlap=False, frozen=False))


def test_broadcast_bias_view():
    """Repeated addresses in a zero-stride bias view receive summed adjoints."""
    w = make_stack_case(203)
    for layer in w["model"]["layers"]:
        for expert in layer["experts"]:
            expert["bias"]["strides"] = [0]
    check(w)


def test_partial_shared_view_offsets():
    """Partially overlapping expert views accumulate on their actual storage coordinates."""
    w = make_stack_case(204, shared=False)
    for layer in w["model"]["layers"]:
        layer["experts"][1] = copy.deepcopy(layer["experts"][0])
        layer["experts"][1]["up"]["offset"] += 1
        layer["experts"][1]["down"]["offset"] += 1
    check(w)


def test_frozen_downstream_blocks_still_propagate():
    """Freezing a downstream layer does not detach its input Jacobian."""
    w = make_stack_case(205, shared=False, overlap=False, frozen=False)
    last = w["model"]["layers"][-1]
    ids = {last["norm"]["block"], last["router"]["block"]}
    ids.update(expert[name]["block"] for expert in last["experts"] for name in ("up", "bias", "down"))
    for i in ids:
        w["blocks"][i]["trainable"] = False
    check(w)


def test_all_storage_frozen():
    """Losses remain defined with no trainable storage and a zero global norm."""
    w = make_stack_case(206)
    for block in w["blocks"]:
        block["trainable"] = False
    expected = reference_stack(w)
    assert all(s["grad_norm"] == 0.0 and s["update_applied"] for s in expected["steps"])
    close(run_cli(w), expected)


def test_auxiliary_loss_reaches_prior_experts():
    """A later router's global balancing objective trains earlier expert outputs."""
    w = make_stack_case(207, shared=False, overlap=False, depth=3, windows=2)
    w["model"]["readout"] = [0.0] * w["model"]["dim"]
    for layer in w["model"]["layers"]:
        layer["aux_coef"] = 0.0
        layer["z_coef"] = 0.0
    w["model"]["layers"][-1]["aux_coef"] = 0.45
    check(w)


def test_z_loss_reaches_prior_norms():
    """Later z-loss pullbacks include the complete earlier normalization chain."""
    w = make_stack_case(208, shared=False, depth=4)
    w["model"]["readout"] = [0.0] * w["model"]["dim"]
    for layer in w["model"]["layers"]:
        layer["aux_coef"] = layer["z_coef"] = 0.0
    w["model"]["layers"][-1]["z_coef"] = 0.3
    check(w)


def test_uneven_window_denominator():
    """Microbatch size and weight differences must not produce a mean of means."""
    w = make_stack_case(209)
    for window in w["windows"]:
        for mi, mb in enumerate(window["microbatches"]):
            for token in mb["tokens"]:
                token["weight"] *= [0.02, 11.0, 1.0][mi]
    check(w)


def test_empty_microbatch_and_empty_window():
    """Empty routing reports coexist with normal zero-gradient optimizer transactions."""
    w = make_stack_case(210, windows=3)
    for mb in w["windows"][0]["microbatches"]:
        for token in mb["tokens"]:
            token["mask"] = False
    for token in w["windows"][1]["microbatches"][0]["tokens"]:
        token["mask"] = False
    w["windows"][2]["microbatches"] = []
    w["windows"][2]["events"] = [{"op": op} for op in ("prepare", "restart", "flush", "flush")]
    check(w)


@pytest.mark.parametrize("dropout", [0.0, 0.6, 1.0])
def test_dropout_edge_cases(dropout):
    """Logical RNG identity is stable even at zero and complete dropout."""
    check(make_stack_case(220, dropout=dropout, depth=3, windows=2))


def test_unconstrained_capacity():
    """Capacity-free admission still retains differentiable accepted gates."""
    check(make_stack_case(221, pressure=False, depth=4))


def test_all_experts_selected():
    """Full top-k selection retains capacity and post-drop gate normalization."""
    w = make_stack_case(222, E=4)
    for layer in w["model"]["layers"]:
        layer["top_k"] = len(layer["experts"])
    check(w)


def test_topk_one_chain():
    """Top-k one removes the task gate derivative but not router auxiliary derivatives."""
    w = make_stack_case(223)
    for layer in w["model"]["layers"]:
        layer["top_k"] = 1
    check(w)


def test_storage_padding_takes_adamw_step():
    """Unreferenced trainable coordinates keep their momentum and decay semantics."""
    w = make_stack_case(224)
    expected = reference_stack(w)
    assert expected["final"]["blocks"][-1]["step"] > w["blocks"][-1]["step"]
    close(run_cli(w), expected)


def test_rmsnorm_radial_jacobian():
    """RMSNorm input derivatives include the shared radial term at small epsilon."""
    check(make_stack_case(225, norm_eps=1e-4, depth=5, windows=2))


def test_overflow_then_recovery():
    """Overflow backoff preserves Adam, EMA and scheduler state until a successful window."""
    w = make_stack_case(226, windows=10)
    w["loss_scaler"]["scale"] = 2.0e6
    expected = reference_stack(w)
    assert any(s["overflow"] for s in expected["steps"])
    assert any(s["update_applied"] for s in expected["steps"])
    close(run_cli(w), expected)


def test_scaler_growth_and_scheduler_wrap():
    """Multiple successful windows update each state cursor only once despite retries."""
    check(make_stack_case(227, windows=9, pressure=False))


def test_public_api_all_receipts():
    """Direct API calls return every reference report and preserve caller ownership."""
    w = make_stack_case(230, windows=3)
    actual = api_chunk(w, journal(w))
    assert_receipts(actual, reference_stack(w))


@pytest.mark.parametrize("boundary", [0, 1, 3, 6, 11, 15, 22])
def test_process_boundary_checkpoint(boundary):
    """A portable checkpoint survives fresh interpreter recovery at different event cuts."""
    w = make_stack_case(231, windows=3)
    events = journal(w)
    cut = min(boundary, len(events))
    first = api_chunk(w, events[:cut])
    second = api_chunk(w, events[cut:], first)
    assert_receipts(second, reference_stack(w))


def test_prepared_state_does_not_escape_checkpoint():
    """Checkpointing a prepared update keeps the public pre-commit state unchanged."""
    w = make_stack_case(232, windows=2)
    events = journal(w)
    cut = next(i + 1 for i, (_, e) in enumerate(events) if e["op"] == "prepare")
    first = api_chunk(w, events[:cut])
    initial = {k: w[k] for k in ("blocks", "model", "scheduler", "loss_scaler")}
    close(first["state"], initial)
    second = api_chunk(w, events[cut:], first)
    assert_receipts(second, reference_stack(w))


def test_recovery_after_committed_response_loss():
    """A retried commit after a process restart returns its original report exactly once."""
    w = make_stack_case(233)
    events = journal(w)
    cut = next(i + 1 for i, (_, e) in enumerate(events) if e["op"] == "flush")
    first = api_chunk(w, events[:cut])
    second = api_chunk(w, events[cut:], first)
    assert_receipts(second, reference_stack(w))


def test_old_event_retry_after_later_updates():
    """Old forward and flush receipts do not rematerialize with newer parameters."""
    w = make_stack_case(234)
    events = journal(w)
    first = api_chunk(w, events)
    old = w["windows"][0]
    retries = [(old["id"], {"op": "forward", "microbatch_id": old["microbatches"][0]["id"]}),
               (old["id"], {"op": "backward", "microbatch_id": old["microbatches"][0]["id"]}),
               (old["id"], {"op": "prepare"}), (old["id"], {"op": "flush"})]
    second = api_chunk(w, retries, first)
    assert_receipts(second, reference_stack(w))


def test_recovery_with_changed_worker_placement():
    """Physical worker labels may change across restore without changing training results."""
    w = make_stack_case(235)
    events = journal(w)
    first = api_chunk(w, events[:7])
    relayout = copy.deepcopy(w)
    for layer in relayout["model"]["layers"]:
        for expert in layer["experts"]:
            expert["home_rank"] = (expert["home_rank"] * 3 + 1) % 11
    for window in relayout["windows"]:
        for mb in window["microbatches"]:
            mb["tokens"].reverse()
            for token in mb["tokens"]:
                token["rank"] = (token["rank"] + 7) % 13
    second = api_chunk(relayout, events[7:], first)
    assert_receipts(second, reference_stack(relayout))


def test_event_schedule_invariance():
    """Delayed backward calls and late global loads preserve the same mathematical transaction."""
    a = make_stack_case(236)
    b = copy.deepcopy(a)
    for window in b["windows"]:
        ids = [mb["id"] for mb in window["microbatches"]]
        window["events"] = ([{"op": "forward", "microbatch_id": i} for i in ids]
                            + [{"op": "backward", "microbatch_id": i} for i in reversed(ids)]
                            + [{"op": "flush"}])
    close(run_cli(a), reference_stack(a))
    close(run_cli(b), reference_stack(a))


def test_rng_same_uid_different_layers_and_microbatches():
    """RNG coordinates distinguish logical layers and microbatches even when sample IDs repeat."""
    w = make_stack_case(237)
    for window in w["windows"]:
        for mb in window["microbatches"]:
            for token in mb["tokens"]:
                token["uid"] = f"sample-{token['ordinal']}"
    check(w)


def test_raw_overflow_survives_checkpoint_and_cancellation():
    """A saved local overflow cannot disappear when later microbatch gradients cancel."""
    from stack_cases import make_stack_cancel_case
    w = make_stack_cancel_case()
    expected = reference_stack(w)
    first = expected["steps"][0]
    assert first["overflow"] and first["grad_norm"] < 1e-12 and first["scaled_grad_peak"] > 65504
    assert expected["steps"][1]["update_applied"]
    events = journal(w)
    before = api_chunk(w, events[:2])
    after = api_chunk(w, events[2:], before)
    assert_receipts(after, expected)


def test_auxiliary_only_window_overflow():
    """Window-global auxiliary gradients affect final AMP status but not local status."""
    w = make_stack_case(240, windows=2)
    w["model"]["readout"] = [0.0] * w["model"]["dim"]
    for layer in w["model"]["layers"]:
        layer["z_coef"] = 0.0
        layer["aux_coef"] = 1.0e5
    expected = reference_stack(w)
    assert any(s["overflow"] for s in expected["steps"])
    close(run_cli(w), expected)
