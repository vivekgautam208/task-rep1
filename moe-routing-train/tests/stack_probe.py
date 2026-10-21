"""Public-API driver executed only in the unprivileged candidate subprocess."""
import copy
import json
import sys
sys.path.insert(0, "/app")
from moe_runtime.engine import Session


def apply(session, window_id, event):
    op = event["op"]
    if op == "forward":
        return session.forward(window_id, event["microbatch_id"])
    if op == "backward":
        return session.backward(window_id, event["microbatch_id"])
    if op in ("prepare", "ascend"):
        before = copy.deepcopy(session.snapshot())
        report = getattr(session, op)(window_id)
        assert before == session.snapshot(), "prepare mutated optimizer state"
        return report
    if op == "flush":
        return session.flush(window_id)
    if op == "restart":
        return None
    raise ValueError(op)


def main():
    request = json.load(open(sys.argv[1], encoding="utf-8"))
    workload = request["workload"]
    original = copy.deepcopy(workload)
    image = request.get("image")
    session = Session.restore(workload, image) if image is not None else Session(workload)
    if image is not None:
        # Destroy borrowed input after restore: caller ownership must be independent.
        image.clear()
    receipts = request.get("receipts", [])
    snapshots = request.get("snapshots", [])
    for entry in request["events"]:
        wid, event = entry
        if event["op"] == "restart":
            snapshot = copy.deepcopy(session.snapshot())
            payload = json.loads(json.dumps(session.checkpoint(), allow_nan=False))
            session = Session.restore(workload, payload)
            payload.clear()
            assert session.snapshot() == snapshot
        else:
            value = apply(session, wid, event)
            receipts.append({"window_id": wid, "event": event, "value": copy.deepcopy(value)})
            if isinstance(value, dict):
                value.clear()
        state_copy = session.snapshot()
        snapshots.append(copy.deepcopy(state_copy))
        state_copy.clear()
    assert workload == original, "input workload was mutated"
    checkpoint = session.checkpoint()
    # A JSON object is the only checkpoint representation requirement.
    assert isinstance(checkpoint, dict)
    serialized = json.dumps(checkpoint, allow_nan=False)
    checkpoint.clear()
    result = {"state": session.snapshot(), "image": json.loads(serialized),
              "receipts": receipts, "snapshots": snapshots}
    with open(sys.argv[2], "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)


if __name__ == "__main__":
    main()
