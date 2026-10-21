"""Outcome verification of cross-microbatch adjoints and temporary optimizer states."""
import copy
import pytest
from coupled_cases import make_coupled_case, rearrange
from stack_cases import make_stack_cancel_case
from stack_reference import reference_stack, evaluate_window, probe_state, gradient_status
from test_outputs import check, close, run_cli
from test_stack import api_chunk, journal


def _receipt_check(actual, expected):
    """Public base-forward receipts differ from the perturbed descent reports."""
    steps = {str(s["window_id"]): s for s in expected["steps"]}
    for receipt in actual["receipts"]:
        step = steps[str(receipt["window_id"])]
        op = receipt["event"]["op"]
        if op == "forward":
            base = step.get("sharpness", {}).get("base", step)
            target = next(m for m in base["microbatches"]
                          if str(m["microbatch_id"]) == str(receipt["event"]["microbatch_id"]))
        elif op == "backward":
            target = None
        elif op == "ascend":
            target = step["sharpness"]["ascent"]
        else:
            target = step
        close(receipt["value"], target)
    close(actual["state"], expected["final"])


def _state(workload):
    """Build the initial numerical state for independent reference probes."""
    return {k: copy.deepcopy(workload[k]) for k in ("blocks", "model", "scheduler", "loss_scaler")}


def _plain_journal(workload):
    """Replace retries with a legal minimal journal, retaining the same training data."""
    for window in workload["windows"]:
        window["events"] = [e for mb in window["microbatches"] for e in (
            {"op": "forward", "microbatch_id": mb["id"]},
            {"op": "backward", "microbatch_id": mb["id"]})] + [{"op": "prepare"}, {"op": "flush"}]


def test_contrastive_window_without_sharpness():
    """Global contrastive gradients train the ordinary accumulated optimizer too."""
    check(make_coupled_case(8101, sharpness=False))


def test_sharpness_without_contrastive_head():
    """The two-pass optimizer is independent of enabling the auxiliary representation head."""
    check(make_coupled_case(8102, contrastive=False, adaptive=False))


def test_representation_only_objective():
    """Removing scalar readout losses isolates the complete contrastive representation pullback."""
    w = make_coupled_case(8103, shared=False, overlap=False)
    w["model"]["readout"] = [0.0] * w["model"]["dim"]
    for layer in w["model"]["layers"]:
        layer["aux_coef"] = layer["z_coef"] = 0.0
    expected = reference_stack(w)
    assert expected["steps"][0]["sharpness"]["ascent"]["base_grad_norm"] > 0.01
    close(run_cli(w), expected)


def test_only_remote_positive_examples():
    """Every positive partner can live in a different microbatch from its anchor."""
    w = make_coupled_case(8104)
    for window in w["windows"]:
        for mb in window["microbatches"]:
            for t in mb["tokens"]:
                t["mask"] = True
                t["class_id"] = str(t["ordinal"])
    check(w)


def test_singletons_are_negative_bank_members():
    """An example without a positive partner still receives candidate-side adjoints."""
    w = make_coupled_case(8105, shared=False, depth=4)
    for window in w["windows"]:
        all_tokens = [t for m in window["microbatches"] for t in m["tokens"]]
        for i, token in enumerate(all_tokens):
            token.update(mask=True, class_id="paired" if i < 3 else f"singleton-{i}")
    check(w)


def test_zero_weight_negative_example():
    """A zero-weight anchor remains a differentiable negative for other anchors."""
    w = make_coupled_case(8106, coefficient=2.0)
    for window in w["windows"]:
        for mi, mb in enumerate(window["microbatches"]):
            for t in mb["tokens"]:
                t["mask"] = True
                if mi == 0:
                    t["weight"] = 0.0
                    t["class_id"] = f"zero-{t['ordinal']}"
    check(w)


def test_no_positive_pairs():
    """No eligible anchors gives zero contrastive loss without disabling other objectives."""
    w = make_coupled_case(8107)
    for window in w["windows"]:
        for mi, mb in enumerate(window["microbatches"]):
            for t in mb["tokens"]:
                t["class_id"] = f"{mi}:{t['ordinal']}"
    expected = reference_stack(w)
    assert all(s["contrastive_loss"] == 0.0 for s in expected["steps"])
    close(run_cli(w), expected)


def test_uid_is_not_global_contrastive_row_identity():
    """RNG UIDs reused in different microbatches must not remove cross-microbatch pairs."""
    w = make_coupled_case(8108)
    for window in w["windows"]:
        for mb in window["microbatches"]:
            for token in mb["tokens"]:
                token["uid"] = f"view-{token['ordinal']}"
                token["class_id"] = str(token["ordinal"] % 2)
    check(w)


def test_anchor_mass_is_not_total_token_mass():
    """High-weight singleton negatives cannot dilute the eligible-anchor denominator."""
    w = make_coupled_case(8109, sharpness=False)
    for window in w["windows"]:
        for mi, mb in enumerate(window["microbatches"]):
            for t in mb["tokens"]:
                t.update(mask=True, class_id="paired" if mi else f"unique-{t['ordinal']}")
                t["weight"] = 31.0 if mi == 0 else 0.13 + t["ordinal"]
    check(w)


@pytest.mark.parametrize("temperature", [0.017, 0.11, 2.3])
def test_temperature_and_logsumexp(temperature):
    """Both loss values and two-sided adjoints remain stable across logit scales."""
    check(make_coupled_case(8110, temperature=temperature, windows=2))


def test_representation_normalization_radial_term():
    """Smoothed L2 normalization cannot be differentiated as a constant scale."""
    w = make_coupled_case(8111, D=2, coefficient=2.8)
    w["model"]["contrastive"]["normalize_epsilon"] = 0.6
    check(w)


def test_scalar_representation_with_nonzero_smoothing():
    """One-dimensional normalized embeddings still have a nonzero derivative with smoothing."""
    check(make_coupled_case(8112, D=1, H=3, E=4, depth=4, coefficient=1.9))


def test_zero_radius_matches_unperturbed_training():
    """A zero-radius probe leaves the optimizer trajectory equal to ordinary training."""
    w = make_coupled_case(8113, radius=0.0)
    ordinary = copy.deepcopy(w)
    ordinary["config"].pop("sharpness")
    _plain_journal(ordinary)
    a, b = run_cli(w), run_cli(ordinary)
    close(a, reference_stack(w))
    close(b, reference_stack(ordinary))
    close(a["final"], b["final"])


@pytest.mark.parametrize("adaptive", [False, True])
def test_unique_storage_perturbation_metric(adaptive):
    """SAM and ASAM perturb unique storage rather than each overlapping logical view."""
    check(make_coupled_case(8114, adaptive=adaptive, depth=5, radius=0.41))


def test_zero_stride_sharpness_views():
    """Broadcast aliases coalesce before defining the perturbation and its normalization."""
    w = make_coupled_case(8115)
    for layer in w["model"]["layers"]:
        for expert in layer["experts"]:
            expert["bias"]["strides"] = [0]
    check(w)


def test_negative_stride_views():
    """Reversed physical views use their real coordinates in both optimizer passes."""
    w = make_coupled_case(8116)
    for layer in w["model"]["layers"]:
        for expert in layer["experts"]:
            up = expert["up"]
            up["offset"] += (up["shape"][0] - 1) * up["strides"][0]
            up["strides"][0] *= -1
    check(w)


def test_adaptive_zero_coordinates_and_eta():
    """The stabilizing adaptive metric remains meaningful at zero-valued coordinates."""
    w = make_coupled_case(8117, adaptive=True)
    w["config"]["sharpness"]["eta"] = 0.24
    for block in w["blocks"]:
        for j in range(0, len(block["weight"]), 3):
            block["weight"][j] = 0.0
    check(w)


def test_all_frozen_probe_is_zero():
    """Frozen storage is excluded from perturbations without disconnecting the forward model."""
    w = make_coupled_case(8118)
    for block in w["blocks"]:
        block["trainable"] = False
    expected = reference_stack(w)
    assert all(s["sharpness"]["ascent"]["perturbation_norm"] == 0.0 for s in expected["steps"])
    close(run_cli(w), expected)


def test_padding_momentum_is_not_a_probe_gradient():
    """An unused momentum-bearing coordinate is not perturbed but still takes its AdamW step."""
    w = make_coupled_case(8119, radius=0.4, graph=False)
    state = _state(w)
    probe, _, _, _ = probe_state(state, w["config"], w["windows"][0])
    assert probe["blocks"][-1]["weight"] == state["blocks"][-1]["weight"]
    expected = reference_stack(w)
    assert expected["final"]["blocks"][-1]["weight"] != state["blocks"][-1]["weight"]
    close(run_cli(w), expected)


def test_routing_can_change_at_the_perturbed_state():
    """A real probe crosses admission boundaries and requires fresh routing and window loads."""
    w = make_coupled_case(8120, adaptive=False, radius=0.85, windows=2)
    expected = reference_stack(w)
    def routes(stats):
        return [r["accepted_experts"] for m in stats["microbatches"] for l in m["layers"] for r in l["routes"]]
    assert any(routes(s) != routes(s["sharpness"]["base"]) for s in expected["steps"])
    close(run_cli(w), expected)


def test_common_random_numbers_across_probe():
    """Recomputation retains stochastic identity despite new parameters and changed admission."""
    check(make_coupled_case(8121, dropout=0.58, radius=0.51, windows=4))


def test_base_local_overflow_survives_cancellation():
    """Canceled base gradients cannot authorize a sharpness descent after local AMP overflow."""
    w = make_stack_cancel_case()
    w["config"]["sharpness"] = {"radius": 0.3, "adaptive": True, "eta": 0.01, "epsilon": 1e-12}
    expected = reference_stack(w)
    assert expected["steps"][0]["sharpness"]["ascent"]["base_overflow"]
    assert not expected["steps"][0]["sharpness"]["descent_evaluated"]
    close(run_cli(w), expected)


def test_descent_only_overflow_rolls_back_perturbation():
    """A finite ascent followed by overflowing descent rolls back weights and optimizer state."""
    w = make_coupled_case(8122, windows=1, radius=0.55, adaptive=False)
    w["loss_scaler"]["scale"] = 1.0
    state = _state(w)
    perturbed, _, _, ascent = probe_state(state, w["config"], w["windows"][0])
    _, g, local = evaluate_window(perturbed, w["config"], w["windows"][0])
    second_peak, _ = gradient_status(perturbed, g, local)
    assert second_peak > ascent["base_scaled_grad_peak"] * 1.001
    w["loss_scaler"]["scale"] = 65504.0 / ((second_peak + ascent["base_scaled_grad_peak"]) / 2.0)
    expected = reference_stack(w)
    assert not expected["steps"][0]["sharpness"]["ascent"]["base_overflow"]
    assert expected["steps"][0]["overflow"]
    assert expected["final"]["blocks"] == w["blocks"]
    close(run_cli(w), expected)


def test_window_contrastive_overflow_with_finite_local_backwards():
    """A global-only objective can overflow even when every local raw backward is zero."""
    w = make_coupled_case(8123, windows=1, coefficient=9.0)
    w["model"]["readout"] = [0.0] * w["model"]["dim"]
    for layer in w["model"]["layers"]:
        layer["aux_coef"] = layer["z_coef"] = 0.0
    w["loss_scaler"]["scale"] = 1.0
    _, _, grads, _ = probe_state(_state(w), w["config"], w["windows"][0])
    peak = max(abs(g) for b, group in zip(w["blocks"], grads) if b["trainable"] for g in group)
    w["loss_scaler"]["scale"] = 65504.0 * 1.2 / peak
    expected = reference_stack(w)
    assert expected["steps"][0]["overflow"]
    close(run_cli(w), expected)


def test_public_receipts_for_both_parameter_states():
    """Forward, ascent and descent reports retain their different state meanings."""
    w = make_coupled_case(8124)
    _receipt_check(api_chunk(w, journal(w)), reference_stack(w))


@pytest.mark.parametrize("boundary", [0, 1, 4, 8, 12, 15, 18, 24])
def test_coupled_fresh_process_cuts(boundary):
    """A checkpoint preserves coupled work across independently launched interpreter processes."""
    w = make_coupled_case(8125, windows=2)
    events = journal(w)
    first = api_chunk(w, events[:boundary])
    second = api_chunk(w, events[boundary:], first)
    _receipt_check(second, reference_stack(w))


def test_checkpoint_immediately_after_ascent():
    """A durable sharpness probe must not expose perturbed values through snapshot."""
    w = make_coupled_case(8126)
    events = journal(w)
    cut = next(i + 1 for i, (_, e) in enumerate(events) if e["op"] == "ascend")
    first = api_chunk(w, events[:cut])
    close(first["state"], _state(w))
    _receipt_check(api_chunk(w, events[cut:], first), reference_stack(w))


def test_prepared_update_is_not_committed_by_checkpoint():
    """Preparing a sharpness update neither commits its state nor loses its durable receipt."""
    w = make_coupled_case(8127)
    events = journal(w)
    cut = next(i + 1 for i, (_, e) in enumerate(events) if e["op"] == "prepare")
    first = api_chunk(w, events[:cut])
    close(first["state"], _state(w))
    _receipt_check(api_chunk(w, events[cut:], first), reference_stack(w))


def test_old_ascent_retry_after_later_commits():
    """An acknowledged ascent receipt remains retrievable after the next optimizer window."""
    w = make_coupled_case(8128, windows=4)
    events = journal(w)
    wid = w["windows"][0]["id"]
    mid = w["windows"][0]["microbatches"][0]["id"]
    first = api_chunk(w, events)
    retries = [(wid, {"op": "ascend"}), (wid, {"op": "prepare"}), (wid, {"op": "flush"}),
               (wid, {"op": "forward", "microbatch_id": mid}), (wid, {"op": "backward", "microbatch_id": mid})]
    _receipt_check(api_chunk(w, retries, first), reference_stack(w))


def test_rank_relayout_during_durable_probe():
    """Relayout changes placement metadata without changing a checkpointed probe's mathematics."""
    w = make_coupled_case(8129)
    events = journal(w)
    cut = next(i + 1 for i, (_, e) in enumerate(events) if e["op"] == "ascend")
    first = api_chunk(w, events[:cut])
    changed = rearrange(w)
    _receipt_check(api_chunk(changed, events[cut:], first), reference_stack(changed))


def test_implicit_ascent_and_minimal_schedule():
    """Prepare can complete a sharpness window without a separately recorded ascent event."""
    w = make_coupled_case(8130)
    _plain_journal(w)
    check(w)


def test_microbatch_partition_is_not_a_contrastive_boundary():
    """With stochastic routing disabled and ample capacity, repartitioning preserves the objective."""
    w = make_coupled_case(8131, pressure=False, dropout=0.0, windows=2)
    for layer in w["model"]["layers"]:
        layer["router_jitter"] = 0.0
    split = copy.deepcopy(w)
    for window in split["windows"]:
        parts = []
        for mb in window["microbatches"]:
            for j in range(2):
                parts.append({"id": f"{mb['id']}-part-{j}", "tokens": mb["tokens"][j::2]})
        window["microbatches"] = parts
    _plain_journal(split)
    a, b = run_cli(w), run_cli(split)
    close(a, reference_stack(w))
    close(b, reference_stack(split))
    close(a["final"], b["final"])


def test_empty_sharpness_window():
    """Empty objectives preserve zero-gradient optimizer and scaler behavior in both passes."""
    w = make_coupled_case(8132, windows=2)
    for token in w["windows"][0]["microbatches"][0]["tokens"]:
        token["mask"] = False
    w["windows"][1]["microbatches"] = []
    _plain_journal(w)
    check(w)


def test_masked_values_cannot_enter_the_negative_bank():
    """Masked examples are absent from scalar losses and global contrastive candidates."""
    w = make_coupled_case(8133)
    for window in w["windows"]:
        for mb in window["microbatches"]:
            for t in mb["tokens"]:
                if not t["mask"]:
                    t["x"] = [1.0e100] * w["model"]["dim"]
                    t["target"] = -1.0e100
                    t["weight"] = 1.0e100
    check(w)


@pytest.mark.parametrize("shape", [(2, 5, 7, 4), (5, 3, 4, 5), (4, 6, 6, 6)])
def test_composed_rectangular_training(shape):
    """Non-square, deep, shared models require all coupled pullbacks and replay guarantees together."""
    D, H, E, depth = shape
    check(rearrange(make_coupled_case(8134 + D, D=D, H=H, E=E, depth=depth, windows=2)))


def test_global_weight_rescaling_preserves_final_update():
    """Uniform token-weight scaling does not change either normalized objective or perturbation."""
    w = make_coupled_case(8139, windows=2)
    w["loss_scaler"]["scale"] = 0.1
    changed = copy.deepcopy(w)
    for window in changed["windows"]:
        for mb in window["microbatches"]:
            for token in mb["tokens"]:
                token["weight"] *= 7.0
    a, b = run_cli(w), run_cli(changed)
    close(a, reference_stack(w))
    close(b, reference_stack(changed))
    close(a["final"], b["final"])



def test_graph_fanout_and_trainable_joins():
    """Every consumer contributes to shared ancestors and softmax join parameters."""
    check(make_coupled_case(8201, depth=6, windows=2, shared=True))


def test_auxiliary_loss_on_a_bypassed_branch():
    """A layer bypassed by the final output still differentiates its own routing objective."""
    w = make_coupled_case(8202, depth=4, shared=False, overlap=False, windows=2)
    w["model"]["layers"][-1]["sources"] = [1, 0]
    w["model"]["layers"][-1]["mix"]["shape"] = [2]
    w["model"]["layers"][2]["aux_coef"] = 1.4
    w["model"]["layers"][2]["z_coef"] = 0.8
    check(w)


def test_repeated_graph_ports_accumulate():
    """Two ports referring to one node add their adjoints instead of overwriting them."""
    w = make_coupled_case(8203, depth=5, windows=2)
    w["model"]["layers"][2]["sources"] = [0, 0, 2]
    w["model"]["layers"][4]["sources"] = [2, 2, 0]
    check(w)


def test_join_parameter_aliases_router_storage():
    """A join and a router can contribute different derivatives to one physical block."""
    w = make_coupled_case(8204, depth=4, windows=2)
    router = w["model"]["layers"][0]["router"]
    for layer in w["model"]["layers"][1:]:
        layer["mix"] = {"block": router["block"], "offset": router["offset"],
                        "shape": [len(layer["sources"])], "strides": [1]}
    check(w)


def test_broadcast_join_logits():
    """Broadcast logits have equal gate values but still require coalescing the softmax pullback."""
    w = make_coupled_case(8205, depth=5, windows=2)
    for layer in w["model"]["layers"][1:]:
        layer["mix"]["strides"] = [0]
    check(w)


def test_frozen_join_keeps_ancestor_gradients():
    """Freezing join parameters must not sever gradient paths to earlier residual nodes."""
    w = make_coupled_case(8206, depth=5, windows=2)
    for layer in w["model"]["layers"][1:]:
        w["blocks"][layer["mix"]["block"]]["trainable"] = False
    check(w)


def test_only_join_parameters_train():
    """Join-only adaptation retains a nonzero perturbation through otherwise frozen layers."""
    w = make_coupled_case(8207, depth=4, windows=2)
    for block in w["blocks"]:
        block["trainable"] = False
    for layer in w["model"]["layers"][1:]:
        w["blocks"][layer["mix"]["block"]]["trainable"] = True
    expected = reference_stack(w)
    assert expected["steps"][0]["sharpness"]["ascent"]["base_grad_norm"] > 1e-5
    close(run_cli(w), expected)


def test_join_port_permutation_is_a_reparameterization():
    """Reversing matching source and logit views preserves both physical-state trajectories."""
    w = make_coupled_case(8208, depth=5, windows=2)
    changed = copy.deepcopy(w)
    for layer in changed["model"]["layers"][1:]:
        layer["sources"].reverse()
        view = layer["mix"]
        view["offset"] += (view["shape"][0] - 1) * view["strides"][0]
        view["strides"][0] *= -1
    a, b = run_cli(w), run_cli(changed)
    close(a, reference_stack(w))
    close(b, reference_stack(changed))
    close(a["final"]["blocks"], b["final"]["blocks"])
