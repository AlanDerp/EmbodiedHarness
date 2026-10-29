"""Run on the System 2 device; use a manifest generated from current evidence."""

import argparse
import json
import os
from pathlib import Path

from embodied_harness.rpc import RpcClient
from embodied_harness.wire import encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    client = RpcClient(args.url, os.environ["EMBODIED_SYSTEM2_TOKEN"], timeout_s=240)
    if args.stop:
        result = client.call("stop")
    elif args.manifest:
        result = client.call("delegate", manifest=json.loads(Path(args.manifest).read_text()))
    else:
        result = client.call("capabilities")
    print(json.dumps(encode(result), indent=2))


if __name__ == "__main__":
    main()
