#!/usr/bin/env python3
import json, sys
from moe_runtime.engine import run


def main():
    if len(sys.argv)!=3:
        raise SystemExit("usage: train_moe.py <workload_json> <output_json>")
    with open(sys.argv[1],"r",encoding="utf-8") as f: workload=json.load(f)
    result=run(workload)
    with open(sys.argv[2],"w",encoding="utf-8") as f: json.dump(result,f,sort_keys=True,separators=(",",":"),allow_nan=False)

if __name__=="__main__": main()
