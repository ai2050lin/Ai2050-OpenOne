# B36 semantic / answer-code calibration, 2026-11-13

Run: B36-SEMANTIC14-20261113-v1. Results: `tests/glm5/result/b36_semantic14_20261113_v1/report.html`.

This stage stopped at the behavior gate. The48-family expanded Python calibration passes3/9 required cells. No new B36 intervention, hidden-field collection, feature fitting or heldout-response confirmation was performed. The frozen32-unit group is only retained as a pending target in stage-decision.json.

Actual execution used `.venv/Scripts/python.exe -B -X utf8 -u` with the following files, in order. All output paths and caches pass deploy/storage_policy.json onD; minimum20GiB free remains. These are frozen scripts, not a command to rerun them into an existing result directory.

1. `b36_semantic14_materials.py`, `test_b36_semantic14_materials.py`, `b36_semantic14_score.py` (default pilot). Pilot interrupted after448 committed rows; pilot-futility-stop.json proves both frames already could not pass. Do not resume it as if the stop had not occurred.
2. `b36_semantic14_calibrate.py`; score with `--plan interface-plan.json --destination interface-score`.
3. `b36_semantic14_arithmetic.py`; score with `--plan arithmetic-interface-plan.json --destination arithmetic-interface-score`.
4. `b36_semantic14_hf_check.py` (original14B CUDA/HF orchestration and tiny cached-generation audit).
5. `b36_semantic14_fewshot.py`; score with `--plan fewshot-plan.json --destination fewshot-score`.
6. `b36_semantic14_order.py`; score with `--plan order-pilot-plan.json --destination order-pilot-score`.
7. `b36_semantic14_reference4b.py` (different model, separate identity;14B process exited first).
8. `b36_semantic14_python.py`; score with `--plan python-plan.json --destination python-score`.
9. `b36_semantic14_expanded.py`; score with `--plan expanded-plan.json --destination expanded-score`.
10. Repeat the3 material engineering tests into material-tests-final.log; `b36_semantic14_audit.py`; `b36_semantic14_report.py`; `b36_semantic14_delivery.py`; append research records; `b36_semantic14_finalize.py` cleans current scratch and writes the manifest.

Each score command redirected stdout/stderr to the matching `*-score.log` in the result directory. The scorer commits64-row chunks with source/plan hashes. Progress lines include elapsed time, ETA, CUDA allocator and DXGI process shared memory. Original14B weight/tokenizer hashes are in model-identity.json;4B uses its own identity in reference4b-check.json. No quantization, no synthetic results. Attention SDPA, native BF16, length256, batch4; CPU reads original weights, CUDA executes one block at a time. Standard cached generation is restricted to the engineering checks.

For read-only reproduction of the decision, audit.py reconstructs2880 planned material rows (including704 unexecuted original pilot rows), token continuations, independent arithmetic/graph/ranking truth and2176 committed score records. Original snapshots/plans/chunks remain immutable. Reports and audit summaries can be regenerated. Repeating GPU measurements needs a freshD result directory, identical frozen inputs and hashes, and new provenance; never delete original results to permit a rerun. The source zip preserves project dependencies, not model files.

The48 new programs exclude every semantic depth1/2/3 prefix of the original6 arithmetic diagnostic families and duplicate prefixes among the new48. They are still discovery calibration, selected after Python diagnostics. Confirmation and lockbox responses remain unopened. A later confirm plan must exclude all prefixes exposed in this stage, retain failed conditions, freeze all code mappings and separate semantic final correctness, format compliance and stopping. No nominal multiple-view count is an independent family count.

No shared protocol or production UI code changed, so the full protocol regression suite was not rerun.3 material unit tests, full artifact audit, source compilation/archive roundtrip and plot inspection are recorded. HTML links were checked; no browser layout claim is made. No server publication or formal evidence registration occurred.
