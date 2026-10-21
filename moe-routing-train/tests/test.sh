#!/bin/sh
set -eu
mkdir -p /logs/verifier
chmod 700 /logs/verifier
if pytest -q /tests/test_outputs.py /tests/test_stack.py /tests/test_coupled.py --ctrf /logs/verifier/ctrf.json; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
  exit 1
fi
