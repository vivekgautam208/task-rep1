import copy, json, math, os, subprocess, tempfile
from pathlib import Path
import pytest
from case_factory import make_case, make_local_cancel_overflow_case, add_retry_events
from reference_model import reference_run
REL = 2e-09
ABS = 2e-10

def close(a, b, path='root'):
    """Close."""
    if isinstance(b, bool) or b is None or isinstance(b, str) or isinstance(b, int):
        assert type(a) is type(b) and a == b, f'{path}: {a!r} != {b!r}'
        return
    if isinstance(b, float):
        assert isinstance(a, (int, float)) and (not isinstance(a, bool)) and math.isfinite(float(a)) and math.isclose(float(a), b, rel_tol=REL, abs_tol=ABS), f'{path}: {a!r} != {b!r}'
        return
    if isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), f'{path}: list length'
        [close(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b))]
        return
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b), f'{path}: keys {set(a) ^ set(b)}'
        [close(a[k], b[k], f'{path}.{k}') for k in b]
        return
    assert a == b

def run_cli(workload):
    """Run cli."""
    with tempfile.TemporaryDirectory() as td:
        os.chmod(td, 511)
        inp = Path(td) / 'in.json'
        out = Path(td) / 'out.json'
        inp.write_text(json.dumps(workload))
        os.chmod(inp, 420)
        cmd = ['setpriv', '--reuid=65534', '--regid=65534', '--clear-groups', '--inh-caps=-all', 'python3', '/app/train_moe.py', str(inp), str(out)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        assert p.returncode == 0, p.stderr[-3000:]
        return json.loads(out.read_text())

def check(workload):
    """Check."""
    close(run_cli(workload), reference_run(workload))

def test_staged_visible_output():
    """Staged visible output."""
    staged = Path('/app/output.json')
    assert not staged.is_symlink(), 'staged output must be a regular file'
    assert staged.is_file(), '/app/output.json is required'
    try:
        actual = json.loads(staged.read_text())
    except Exception as exc:
        pytest.fail(f'/app/output.json is not valid JSON: {exc}')
    workload = json.load(open('/tests/visible_workload.json'))
    close(actual, reference_run(workload))

def api_replay(workload):
    """Api replay."""
    with tempfile.TemporaryDirectory() as td:
        os.chmod(td, 511)
        inp = Path(td) / 'in.json'
        out = Path(td) / 'out.json'
        probe = Path(td) / 'probe.py'
        inp.write_text(json.dumps(workload))
        os.chmod(inp, 420)
        probe.write_text('import json,sys\nsys.path.insert(0,"/app")\nfrom moe_runtime.engine import Session\nw=json.load(open(sys.argv[1]));s=Session(w);steps=[];flushed={}\nfor window in w["windows"]:\n  for ev in window["events"]:\n    if ev["op"]=="forward": s.forward(window["id"],ev["microbatch_id"])\n    elif ev["op"]=="backward": s.backward(window["id"],ev["microbatch_id"])\n    elif ev["op"]=="ascend": s.ascend(window["id"])\n    elif ev["op"]=="prepare": s.prepare(window["id"])\n    elif ev["op"]=="restart": s=Session.restore(w,json.loads(json.dumps(s.checkpoint(),allow_nan=False)))\n    elif ev["op"]=="flush":\n      r=s.flush(window["id"]);wid=str(window["id"])\n      if wid in flushed: assert r==flushed[wid]\n      else: flushed[wid]=r;steps.append(r)\njson.dump({"steps":steps,"final":s.snapshot()},open(sys.argv[2],"w"),allow_nan=False,sort_keys=True)\n')
        os.chmod(probe, 493)
        p = subprocess.run(['setpriv', '--reuid=65534', '--regid=65534', '--clear-groups', '--inh-caps=-all', 'python3', str(probe), str(inp), str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        assert p.returncode == 0, p.stderr[-3000:]
        return json.loads(out.read_text())

def check_api(workload):
    """Check api."""
    close(api_replay(workload), reference_run(workload))

def test_visible_replay():
    """Visible replay."""
    w = json.load(open('/tests/visible_workload.json'))
    check(w)

def test_visible_public_session_api():
    """Visible public session api."""
    w = json.load(open('/tests/visible_workload.json'))
    check_api(w)

def test_reverse_backward_schedule():
    """Reverse backward schedule."""
    check(make_case(41, schedule_variant=0, token_reorder=True))

def test_interleaved_forward_backward():
    """Interleaved forward backward."""
    check(make_case(42, schedule_variant=1, E=7, P=4, top_k=3))

def test_reverse_forward_schedule():
    """Reverse forward schedule."""
    check(make_case(43, schedule_variant=2, E=9, P=5, D=5, H=4))

def test_shuffled_event_schedule_api():
    """Shuffled event schedule api."""
    check_api(make_case(44, schedule_variant=3, E=8, P=3, top_k=3))

def test_topology_relayout_invariance():
    """Topology relayout invariance."""
    check(make_case(45, rank_relayout=True, schedule_variant=3))

def test_input_order_invariance():
    """Input order invariance."""
    check(make_case(46, token_reorder=True, E=10, P=6, top_k=2))

def test_dense_alias_groups():
    """Dense alias groups."""
    check(make_case(47, E=10, P=2, top_k=3, schedule_variant=2))

def test_no_alias_baseline():
    """No alias baseline."""
    check(make_case(48, E=5, P=5, alias=False, top_k=2))

def test_topk_one_pressure():
    """Topk one pressure."""
    check(make_case(49, E=7, P=4, top_k=1, pressure=True))

def test_topk_four_pressure():
    """Topk four pressure."""
    check(make_case(50, E=9, P=5, top_k=4, pressure=True, D=5, H=5))

def test_no_capacity_drops():
    """No capacity drops."""
    check(make_case(51, pressure=False, E=6, P=3, top_k=2))

def test_heavy_dropout():
    """Heavy dropout."""
    check(make_case(52, dropout=0.55, E=8, P=4, H=5, schedule_variant=3))

def test_no_dropout_still_stateless_jitter():
    """No dropout still stateless jitter."""
    check(make_case(53, dropout=0.0, jitter=0.23, E=8, P=5))

def test_no_jitter_with_dropout():
    """No jitter with dropout."""
    check(make_case(54, dropout=0.35, jitter=0.0, E=7, P=3, schedule_variant=2))

def test_high_jitter_changes_selection():
    """High jitter changes selection."""
    check(make_case(55, jitter=0.65, E=9, P=5, top_k=3))

def test_expert_only_overflow_transaction():
    """Expert only overflow transaction."""
    check(make_case(56, overflow=True, E=8, P=4, top_k=3, D=5, H=6))

def test_overflow_with_aliasing_and_schedule_api():
    """Overflow with aliasing and schedule api."""
    check_api(make_case(57, overflow=True, E=10, P=2, top_k=4, schedule_variant=3))

def test_global_clipping_many_unique_blocks():
    """Global clipping many unique blocks."""
    check(make_case(58, E=9, P=8, top_k=3, D=6, H=4))

def test_scheduler_wraparound_multistep():
    """Scheduler wraparound multistep."""
    check(make_case(59, windows=8, E=7, P=4, schedule_variant=1))

def test_scaler_growth_multistep():
    """Scaler growth multistep."""
    check(make_case(60, windows=7, E=6, P=3, pressure=False))

def test_small_dimension_large_hidden():
    """Small dimension large hidden."""
    check(make_case(61, D=2, H=7, E=7, P=4, top_k=2))

def test_large_dimension_small_hidden():
    """Large dimension small hidden."""
    check(make_case(62, D=8, H=2, E=8, P=5, top_k=3))

def test_uneven_weight_extremes():
    """Uneven weight extremes."""
    check(make_case(63, E=9, P=4, top_k=3, windows=6))

def test_reordered_topology_and_api():
    """Reordered topology and api."""
    check_api(make_case(64, rank_relayout=True, token_reorder=True, schedule_variant=2, E=10, P=3, top_k=4))

def test_long_alias_heavy_recovery():
    """Long alias heavy recovery."""
    check(make_case(65, E=12, P=3, D=5, H=5, top_k=4, windows=9, schedule_variant=3))

def test_long_many_unique_recovery_api():
    """Long many unique recovery api."""
    check_api(make_case(66, E=10, P=9, D=6, H=4, top_k=3, windows=8, schedule_variant=0))

def test_stochastic_reproducibility_same_input():
    """Stochastic reproducibility same input."""
    w = make_case(67, E=8, P=4, top_k=3, schedule_variant=3)
    a = run_cli(w)
    b = run_cli(copy.deepcopy(w))
    close(a, b)

def test_schedule_only_metamorphic_equivalence():
    """Schedule only metamorphic equivalence."""
    a = make_case(68, E=8, P=4, top_k=3, schedule_variant=0)
    b = copy.deepcopy(a)
    for w in b['windows']:
        mids = [m['id'] for m in w['microbatches']]
        w['events'] = [*({'op': 'forward', 'microbatch_id': m} for m in reversed(mids)), *({'op': 'backward', 'microbatch_id': m} for m in mids), {'op': 'flush'}]
    close(run_cli(a), run_cli(b))

def test_rank_only_metamorphic_equivalence():
    """Rank only metamorphic equivalence."""
    a = make_case(69, E=9, P=4, top_k=3)
    b = copy.deepcopy(a)
    for e in b['experts']:
        e['home_rank'] = (e['home_rank'] + 3) % 7
    for w in b['windows']:
        for mb in w['microbatches']:
            for t in mb['tokens']:
                t['rank'] = (t['rank'] + 4) % 7
    x = run_cli(a)
    y = run_cli(b)
    x['final']['experts'] = y['final']['experts']
    close(x, y)

def test_token_order_metamorphic_equivalence():
    """Token order metamorphic equivalence."""
    a = make_case(70, E=8, P=5, top_k=3)
    b = copy.deepcopy(a)
    for w in b['windows']:
        for mb in w['microbatches']:
            mb['tokens'].reverse()
    close(run_cli(a), run_cli(b))

def test_empty_microbatch_inside_window():
    """Empty microbatch inside window."""
    w = make_case(71, E=7, P=4, top_k=2)
    mb = w['windows'][0]['microbatches'][0]
    for t in mb['tokens']:
        t['mask'] = False
    check(w)

def test_all_masked_window_still_updates_zero_grad_blocks():
    """All masked window still updates zero grad blocks."""
    w = make_case(72, E=6, P=3, top_k=2, windows=2)
    for t in w['windows'][0]['microbatches'][0]['tokens']:
        t['mask'] = False
    for mb in w['windows'][0]['microbatches'][1:]:
        for t in mb['tokens']:
            t['mask'] = False
    check(w)

def test_mixed_frozen_shared_blocks():
    """Mixed frozen shared blocks."""
    w = make_case(73, E=10, P=4, top_k=3, schedule_variant=1)
    w['expert_params'][0]['trainable'] = False
    w['expert_params'][1]['trainable'] = True
    check(w)

def test_all_expert_blocks_frozen_router_only_update():
    """All expert blocks frozen router only update."""
    w = make_case(74, E=8, P=4, top_k=3, schedule_variant=3)
    for b in w['expert_params']:
        b['trainable'] = False
    check(w)

def test_frozen_alias_topology_api():
    """Frozen alias topology api."""
    w = make_case(75, E=12, P=3, top_k=4, rank_relayout=True, token_reorder=True, schedule_variant=2)
    w['expert_params'][1]['trainable'] = False
    check_api(w)

def test_global_aux_backward_before_later_forwards():
    """Global aux backward before later forwards."""
    w = make_case(76, E=9, P=5, top_k=3, schedule_variant=1, windows=6)
    check_api(w)

def test_overflow_backoff_then_successful_recovery():
    """Overflow backoff then successful recovery."""
    w = make_case(77, E=9, P=4, D=5, H=4, top_k=3, windows=10, schedule_variant=3)
    w['loss_scaler']['scale'] = 2000000.0
    w['loss_scaler']['growth_tracker'] = 0
    expected = reference_run(w)
    assert any((s['overflow'] for s in expected['steps']))
    assert any((s['update_applied'] for s in expected['steps']))
    close(run_cli(w), expected)

def test_at_least_once_retry_events_cli():
    """At least once retry events cli."""
    w = add_retry_events(make_case(78, E=10, P=3, top_k=4, windows=7, schedule_variant=3, token_reorder=True))
    check(w)

def test_at_least_once_retry_events_api():
    """At least once retry events api."""
    w = add_retry_events(make_case(79, E=9, P=4, top_k=3, windows=6, schedule_variant=1, rank_relayout=True))
    check_api(w)

def test_local_scaled_overflow_survives_later_cancellation():
    """Local scaled overflow survives later cancellation."""
    w = make_local_cancel_overflow_case()
    expected = reference_run(w)
    assert expected['steps'][0]['overflow'] is True
    assert expected['steps'][0]['scaled_grad_peak'] > 65504.0
    assert expected['steps'][0]['grad_norm'] < 1e-12
    check(w)
