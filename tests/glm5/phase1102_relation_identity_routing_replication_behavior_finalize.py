#!/usr/bin/env python3
"""Freeze Phase1102 behavior authorization."""

from __future__ import annotations

import phase1101_relation_identity_routing_behavior_finalize as shared
import phase1102_relation_identity_routing_replication_protocol as protocol


shared.protocol = protocol


if __name__ == "__main__":
    shared.main()
