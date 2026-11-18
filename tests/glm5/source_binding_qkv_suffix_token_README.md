# 中文问句末段的逐 token 替换、屏蔽与恢复

用 Qwen3-14B BF16 和原图谱固定的 16 个头。所有输入末尾固定为 `Answer:`。CPU 读取原始权重，CUDA 按块执行。旧逐条宽编辑答案作固定目标，材料文件同时保存供体、原始 token、问句片段与位置。

## 逐位置定位

每个模板的问句末段含 5–7 个 token。固定第 23、24 或 25 块入口和原来的来源名、实体名或两者干预。基线 H 只在作答位置的指定头替换名字 K/V。单 token 臂在此基础上额外读取供体的同一个问句末段位置，整段臂做正对照。句面文字与 token ID 保持原样。

用第一批 12 个家族在每个模板选一个位置。排序依据是宽编辑确实改变答案的样本上，与 H 相比复现率提高多少，平手用四选一分差，然后取更早位置。选择写入 `selected.json` 后，使用另一批 12 个家族复测。两个家族分组彼此不交叠，但原材料曾用于先前实验。报告另外列出固定词元「动物」和距问号的相对位置。

## 正常答题的干预

所有操作使用原题和原来的正确答案。只改这 16 个头在第 23–27 块从作答位置到指定位置的注意力边，保留其他头及先前位置的计算。

- `mask_best` 屏蔽冻结位置。`mask_control` 屏蔽循环相邻的一个末段 token。`mask_prior` 屏蔽由固定种子选择的较早位置。`mask_t0` 等逐个屏蔽末段位置。
- `mask_all` 屏蔽末段所有位置。`reopen_best` 在整段屏蔽中仅开放冻结位置，`reopen_control` 仅开放相邻对照。软最大注意力会在可见位置重新分配权重。
- `zero_*` 保留注意力候选位置，令指定边携带的 V 为零。`restore_best`、`restore_control` 从 `zero_all` 恢复原生的单位置 V，`restore_all` 全部恢复，并与同形状原生重放逐项核对。

准确率使用模型在四条候选记录中的选择。分差使用正确记录 logit 减三个竞争记录中的最高 logit。另存全词表首选准确率。配对区间按题目家族重抽样。

## 运行与复算

从仓库根目录运行。分片保存在 `tests/glm5/result/source_binding_qkv_followup_v1/suffix-token/`，已完成的分片可以直接续用。`design` 仅用于新结果目录。

```powershell
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token.py design
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token.py scan --split discover
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token_analysis.py scan --split discover
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token.py scan --split confirm
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token_analysis.py scan --split confirm
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token.py natural --split confirm
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token_analysis.py natural --split confirm
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token.py natural --split discover
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token_analysis.py natural --split discover
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_token_audit.py
```

正常答题对照先执行第二批家族，再用第一批材料扩展覆盖。`audit.json` 汇总两批的自然读出变化，逐条核对旧后缀对照、原答题基线、哈希和完整恢复。
