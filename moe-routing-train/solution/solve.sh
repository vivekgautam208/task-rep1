#!/bin/sh
set -eu
cp /solution/train_moe.py /app/train_moe.py
rm -rf /app/moe_runtime
cp -R /solution/moe_runtime /app/moe_runtime
chmod 755 /app/train_moe.py
python3 /app/train_moe.py /app/data/visible_workload.json /app/output.json
