# Rejected Phase 983 sealed chain: orchestrator log encoding

Rejected at 2026-07-18 22:59 America/Chicago before any formal result row was
written.  This directory is evidence only and MUST NOT be admitted for formal
generation or scientific analysis.

## What happened

The first formal Qwen3 child began loading model weights.  Its Transformers
progress stream inherited a Windows console encoding and emitted byte `0xA8`.
The parent orchestrator deliberately used strict UTF-8 decoding and stopped
with `UnicodeDecodeError` rather than replacing evidence bytes silently.

Post-failure checks established:

- no Python model process remained;
- no CUDA compute process remained;
- no runner or orchestrator lock remained;
- the Qwen3 formal output directory contained zero files and zero rows;
- GLM4 and DS7B were never started.

The orchestrator was then changed to set `PYTHONIOENCODING=utf-8` and
`PYTHONUTF8=1` in every runner child.  Because that script is protocol-sealed,
the protocol, engineering qualification, and admission below were archived
together and a new chain is required.

## Rejected lineage

- protocol content SHA-256:
  `bf5a18a946a05a0161232f520aaff5d6ed0b3e5c299f6a5657ed412647e008cf`
- protocol file SHA-256:
  `923cdc0a3cb9c746bb584b5b1695893ffb58b220c383a695a810638a78105500`
- qualification content SHA-256:
  `bbba5b990b6bf427a956504ad2fe46a1badc87abee185aef248cab295b9a4d00`
- admission content SHA-256:
  `9b7f5c70b8577a7a1d155ac6a5382aac643ae8d4844636bac24d5326ff3c87cb`
- rejected orchestrator script seal:
  `c246b2cff7594f3bf10437b962440edc0c9fea28f52fbdfd2364a5d10625c0bd`

This failure is an engineering provenance event, not a model observation and
not evidence for or against the Phase 983 scientific hypothesis.
