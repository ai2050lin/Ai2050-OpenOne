#!/usr/bin/env python3
"""Run Phase1102 behavior replication for one model."""

from __future__ import annotations

import argparse

import phase1101_relation_identity_routing_behavior as shared
import phase1102_relation_identity_routing_replication_protocol as protocol


shared.protocol = protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    shared.run(args.model)


if __name__ == "__main__":
    main()
