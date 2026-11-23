// Read-only excerpts for the static archive browser. Object indices are zero-based within the recorded model snapshot.
window.modelRecords = [
  {
    "id": "p2983-confirm-000-02",
    "phase": 2983,
    "title": "Orion 档案 · 按物理首行找代码",
    "question": "In the active archive, find the CODE on the data line just below BEGIN. Return the code alone, with no punctuation.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "clt_b26_three_cues_v3",
    "caseId": "confirm-000-02",
    "scope": "确认集 · 单题",
    "objects": [
      {
        "kind": "Head",
        "block": 26,
        "index": 36
      }
    ],
    "result": "模型答出 silver。B26/H36 在读到最后一个输入 token 时，对第 4 行的记录注意力最大，题目指定的是第 1 行。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "silver"
      },
      {
        "label": "目标记录",
        "value": "第 1 行"
      },
      {
        "label": "峰值记录注意力",
        "value": "34.1%"
      }
    ],
    "source": "frontend/src/components/v8/threeCueEvidence.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "clt_b26_three_cues_v3 / confirm-000-02",
    "materialSource": "tests/glm5/result/clt_b26_three_cues_v3/confirm-materials.json",
    "artifactSha": "fd5db4808b38ffdfd5fcd25b023ee71f4bb624acd3871a94cd81d5723d3174ba",
    "promptType": "原题问句",
    "readout": "最后输入 token"
  },
  {
    "id": "p2933-counted0-u2429",
    "phase": 2933,
    "title": "数步后读代码 · U2429 的四次读数",
    "question": "Steps: 20,27  \nValue: 27  \nSlot: 1  \nCode:",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "B36-TRANSITION14-20261114-v1",
    "caseId": "t01eb4d20ccb2964:counted0",
    "scope": "单题 · 四个位置",
    "objects": [
      {
        "kind": "MLP",
        "block": 36,
        "index": 2429
      }
    ],
    "result": "同一个 MLP 单元在输入末尾为 2.109375，在最后一步前及 Value 前转为负值，到 Code 前回升至 0.79296875。该条目保存了四处原生激活。",
    "metrics": [
      {
        "label": "输入末尾",
        "value": "2.109375"
      },
      {
        "label": "最后一步前",
        "value": "-0.4765625"
      },
      {
        "label": "Value 前",
        "value": "-0.447265625"
      },
      {
        "label": "Code 前",
        "value": "0.79296875"
      }
    ],
    "source": "frontend/src/components/v8/data/graphObservationExample.json",
    "status": "本地历史摘录",
    "receipt": null,
    "artifactSha": "5778211c68b9e60b45535bdd4426fbed68244da4100d8540488114e35d0c2bdd",
    "sourceKey": "native/confirm-000/native.npz",
    "promptType": "原记录中的生成前缀"
  },
  {
    "id": "p2997-orion-head-block",
    "phase": 2997,
    "title": "Orion 档案 · 阻断首行注意力头",
    "question": "In the active archive, find the CODE on the data line just below BEGIN. Return the code alone, with no punctuation.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "routine-run-t7dfn8_7-1795090345387617100",
    "caseId": "intervention-000-02-target",
    "scope": "单题 · 匹配对照",
    "objects": [
      {
        "kind": "Head",
        "block": 28,
        "index": 21
      }
    ],
    "result": "本题定向阻断后模型仍答 fire。原样、定向阻断和同题随机对照的正确项优势分别为 10.25、10.50、11.125。单题变化方向可与家族平均值不同。",
    "metrics": [
      {
        "label": "原样优势",
        "value": "10.25"
      },
      {
        "label": "阻断后优势",
        "value": "10.5"
      },
      {
        "label": "随机头对照",
        "value": "11.125"
      }
    ],
    "source": "tests/glm5/result/routine/run-t7dfn8_7/questions.json",
    "status": "中心已读回",
    "receipt": "98f5ff92dcf1c17418c9e47708fe84366f35fea74db9bd91320f52ede5d8973e",
    "sourceKey": "case_results.jsonl / intervention-000-02-target",
    "promptType": "原题问句",
    "artifactSha": "a3711933f893b895ea3e127a5213edfc0b7f8bae7692842f5d74f0a97c2b715c",
    "resultArtifact": "case_results.jsonl",
    "receiptSource": "tests/glm5/result/routine/run-t7dfn8_7/server-receipt.json",
    "evidenceStatus": "unreviewed"
  },
  {
    "id": "claude261-u9517-s9-1",
    "phase": 261,
    "title": "香港与巴黎的城市往来 · U9517",
    "question": "A　香港与内地的往来越来越频繁。\nB　巴黎与里昂的往来越来越频繁。",
    "model": "Qwen3-4B",
    "snapshot": "6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad",
    "dims": {
      "layers": 36,
      "heads": 32,
      "width": 9728
    },
    "run": null,
    "caseId": null,
    "scope": "最小句对 · 观察",
    "objects": [
      {
        "kind": "MLP",
        "block": 30,
        "index": 9517
      }
    ],
    "result": "逐句取 U9517 在各 token 上最大的门控投影，再用同份语料的均值和标准差换算。A 减 B 为 +5.219 个标准差。这对句子只提供激活读数。",
    "metrics": [
      {
        "label": "A · 标准分",
        "value": "6.381"
      },
      {
        "label": "B · 标准分",
        "value": "1.162"
      },
      {
        "label": "A − B",
        "value": "+5.219"
      }
    ],
    "source": "research/neurons/goodfire/GF-L30-U9517-local.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "S9.pairs[0]",
    "promptType": "原始最小句对",
    "series": "Claude",
    "artifactSha": "878f2cb3b60428da80ac3160ef30a39e941bf138b025cae5cc59ef21a3a70ccf",
    "snapshotReference": "research/mainline/H0/report_v5.json"
  },
  {
    "id": "p3008-color-base-45",
    "phase": 3008,
    "title": "成熟草莓与消防车是否同色？",
    "question": "Consider the usual visible surface color of each item. First item: a ripe strawberry. Second item: a fire engine in its traditional livery. Are their typical colors the same or different? Answer same or different.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "record-74286d8e-51c4-4704-97f7-ca3bd0599f3c",
    "caseId": "45",
    "scope": "单题 · 全模型生成",
    "objects": [],
    "result": "原样贪心生成的答案是 different，参考答案是 same，判为错误。本条记录提供全模型行为结果，没有记录单个 Head 或 MLP 的读数。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "different"
      },
      {
        "label": "参考答案",
        "value": "same"
      },
      {
        "label": "判定",
        "value": "错误"
      }
    ],
    "source": "tests/glm5/result/color_reuse_v1/greedy/study-package/case_results.jsonl",
    "status": "中心已读回",
    "receipt": "4e62c143a00b1c71cd8c33a8e7175e549d095b621e6d8ea8071d4c4f2b8fed20",
    "condition": "base",
    "sourceKey": "id=45, arm=base",
    "promptType": "原题",
    "artifactSha": "b0559b89b0bc7407d7f0076a4cf71acdd588b53046971fb845cc461f1e1dd1a3",
    "receiptSource": "tests/glm5/result/color_reuse_v1/greedy/study-receipt.json",
    "evidenceStatus": "unreviewed"
  },
  {
    "id": "p2990-confirm-000-00",
    "phase": 2990,
    "title": "Orion 档案 · 按名字找代码",
    "question": "In the active archive, find the CODE in the record named Rerole. Return the code alone, with no punctuation.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "clt_b26_competition_v1",
    "caseId": "confirm-000-00",
    "scope": "确认集 · 单题",
    "objects": [
      {
        "kind": "Head",
        "block": 26,
        "index": 36
      }
    ],
    "result": "模型答出 rice。B26/H36 在读到最后一个输入 token 时，对第 2 行的记录注意力最大，题目指定的是第 2 行。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "rice"
      },
      {
        "label": "目标记录",
        "value": "第 2 行"
      },
      {
        "label": "峰值记录注意力",
        "value": "41.1%"
      }
    ],
    "source": "frontend/src/components/v8/threeCueEvidence.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "clt_b26_competition_v1 / confirm-000-00",
    "materialSource": "tests/glm5/result/clt_b26_competition_v1/confirm-materials.json",
    "artifactSha": "7cabcc7af2b446ab5c91dcc56a8bb6a04426b8ebdc318f672bababde6f63f884",
    "promptType": "原题问句",
    "readout": "最后输入 token"
  },
  {
    "id": "p2997-first-line-cohort",
    "phase": 2997,
    "title": "物理首行 · 注意力头阻断与单元组恢复",
    "question": "阻断 B28/H21 后，答案优势的变化能否用随机头重现？恢复同层的 MLP 单元组能补回多少？",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "routine-run-t7dfn8_7-1795090345387617100",
    "caseId": null,
    "scope": "192 家族 · 运行汇总",
    "objects": [
      {
        "kind": "Head",
        "block": 28,
        "index": 21
      },
      {
        "kind": "MLP",
        "block": 28,
        "index": 17216
      }
    ],
    "result": "定向阻断相对随机头让平均答案优势再降 0.330。单独恢复 MLP 组的平均优势增益为 -0.151。U17216 是冻结的 128 单元之一，组指标不能当成该单元的独立效应。",
    "metrics": [
      {
        "label": "家族数",
        "value": "192"
      },
      {
        "label": "额外优势降幅",
        "value": "0.330"
      },
      {
        "label": "组恢复增益",
        "value": "-0.151"
      }
    ],
    "source": "tests/glm5/result/routine/run-t7dfn8_7/review.json",
    "status": "中心已读回",
    "receipt": "98f5ff92dcf1c17418c9e47708fe84366f35fea74db9bd91320f52ede5d8973e",
    "sourceKey": "conclusion.causal",
    "promptType": "研究问题",
    "artifactSha": "1d2170e729f428e19d6b74a12d8dbdbcda882796c3b7f56a65a0da63a2bfb349",
    "receiptSource": "tests/glm5/result/routine/run-t7dfn8_7/server-receipt.json",
    "evidenceStatus": "unreviewed"
  },
  {
    "id": "claude261-u80-s9-2",
    "phase": 261,
    "title": "两岸关系与健康饮食 · U80",
    "question": "A　两岸关系是大陆和台湾共同关心的话题。\nB　健康饮食是大家共同关心的话题。",
    "model": "Qwen3-4B",
    "snapshot": "6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad",
    "dims": {
      "layers": 36,
      "heads": 32,
      "width": 9728
    },
    "run": null,
    "caseId": null,
    "scope": "最小句对 · 观察",
    "objects": [
      {
        "kind": "MLP",
        "block": 30,
        "index": 80
      }
    ],
    "result": "逐句取 U80 在各 token 上最大的门控投影，再用同份语料的均值和标准差换算。A 减 B 为 -0.320 个标准差。这对句子只提供激活读数。",
    "metrics": [
      {
        "label": "A · 标准分",
        "value": "1.105"
      },
      {
        "label": "B · 标准分",
        "value": "1.425"
      },
      {
        "label": "A − B",
        "value": "-0.320"
      }
    ],
    "source": "research/neurons/goodfire/GF-L30-U80-local.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "S9.pairs[1]",
    "promptType": "原始最小句对",
    "series": "Claude",
    "artifactSha": "1174879499b5cd32915a7a3c3a8f4c0c0ad2cecc6cf1994b15992a527a5efac1",
    "snapshotReference": "research/mainline/H0/report_v5.json"
  },
  {
    "id": "p2983-confirm-000-00",
    "phase": 2983,
    "title": "Orion 档案 · 按名字找代码",
    "question": "In the active archive, find the CODE in the record named Tebaka. Return the code alone, with no punctuation.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "clt_b26_three_cues_v3",
    "caseId": "confirm-000-00",
    "scope": "确认集 · 单题",
    "objects": [
      {
        "kind": "Head",
        "block": 26,
        "index": 36
      }
    ],
    "result": "模型答出 wood。B26/H36 在读到最后一个输入 token 时，对第 2 行的记录注意力最大，题目指定的是第 2 行。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "wood"
      },
      {
        "label": "目标记录",
        "value": "第 2 行"
      },
      {
        "label": "峰值记录注意力",
        "value": "40.5%"
      }
    ],
    "source": "frontend/src/components/v8/threeCueEvidence.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "clt_b26_three_cues_v3 / confirm-000-00",
    "materialSource": "tests/glm5/result/clt_b26_three_cues_v3/confirm-materials.json",
    "artifactSha": "fd5db4808b38ffdfd5fcd25b023ee71f4bb624acd3871a94cd81d5723d3174ba",
    "promptType": "原题问句",
    "readout": "最后输入 token"
  },
  {
    "id": "claude261-u9517-s9-3",
    "phase": 261,
    "title": "过关去香港与出发去学校 · U9517",
    "question": "A　他从深圳过关去了香港。\nB　他从车站出发去了学校。",
    "model": "Qwen3-4B",
    "snapshot": "6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad",
    "dims": {
      "layers": 36,
      "heads": 32,
      "width": 9728
    },
    "run": null,
    "caseId": null,
    "scope": "最小句对 · 观察",
    "objects": [
      {
        "kind": "MLP",
        "block": 30,
        "index": 9517
      }
    ],
    "result": "逐句取 U9517 在各 token 上最大的门控投影，再用同份语料的均值和标准差换算。A 减 B 为 +5.033 个标准差。这对句子只提供激活读数。",
    "metrics": [
      {
        "label": "A · 标准分",
        "value": "6.111"
      },
      {
        "label": "B · 标准分",
        "value": "1.077"
      },
      {
        "label": "A − B",
        "value": "+5.033"
      }
    ],
    "source": "research/neurons/goodfire/GF-L30-U9517-local.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "S9.pairs[2]",
    "promptType": "原始最小句对",
    "series": "Claude",
    "artifactSha": "878f2cb3b60428da80ac3160ef30a39e941bf138b025cae5cc59ef21a3a70ccf",
    "snapshotReference": "research/mainline/H0/report_v5.json"
  },
  {
    "id": "p3008-color-base-47",
    "phase": 3008,
    "title": "成熟草莓与香蕉是否同色？",
    "question": "Consider the usual visible surface color of each item. First item: a ripe strawberry. Second item: a ripe Cavendish banana. Are their typical colors the same or different? Answer same or different.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "record-74286d8e-51c4-4704-97f7-ca3bd0599f3c",
    "caseId": "47",
    "scope": "单题 · 全模型生成",
    "objects": [],
    "result": "原样贪心生成的答案是 different，参考答案是 different，判为正确。本条记录提供全模型行为结果，没有记录单个 Head 或 MLP 的读数。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "different"
      },
      {
        "label": "参考答案",
        "value": "different"
      },
      {
        "label": "判定",
        "value": "正确"
      }
    ],
    "source": "tests/glm5/result/color_reuse_v1/greedy/study-package/case_results.jsonl",
    "status": "中心已读回",
    "receipt": "4e62c143a00b1c71cd8c33a8e7175e549d095b621e6d8ea8071d4c4f2b8fed20",
    "condition": "base",
    "sourceKey": "id=47, arm=base",
    "promptType": "原题",
    "artifactSha": "b0559b89b0bc7407d7f0076a4cf71acdd588b53046971fb845cc461f1e1dd1a3",
    "receiptSource": "tests/glm5/result/color_reuse_v1/greedy/study-receipt.json",
    "evidenceStatus": "unreviewed"
  },
  {
    "id": "p2990-confirm-000-02",
    "phase": 2990,
    "title": "Orion 档案 · 按物理首行找代码",
    "question": "In the active archive, find the CODE on the data line just below BEGIN. Return the code alone, with no punctuation.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "clt_b26_competition_v1",
    "caseId": "confirm-000-02",
    "scope": "确认集 · 单题",
    "objects": [
      {
        "kind": "Head",
        "block": 26,
        "index": 36
      }
    ],
    "result": "模型答出 dog。B26/H36 在读到最后一个输入 token 时，对第 1 行的记录注意力最大，题目指定的是第 1 行。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "dog"
      },
      {
        "label": "目标记录",
        "value": "第 1 行"
      },
      {
        "label": "峰值记录注意力",
        "value": "15.7%"
      }
    ],
    "source": "frontend/src/components/v8/threeCueEvidence.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "clt_b26_competition_v1 / confirm-000-02",
    "materialSource": "tests/glm5/result/clt_b26_competition_v1/confirm-materials.json",
    "artifactSha": "7cabcc7af2b446ab5c91dcc56a8bb6a04426b8ebdc318f672bababde6f63f884",
    "promptType": "原题问句",
    "readout": "最后输入 token"
  },
  {
    "id": "claude261-u80-s9-1",
    "phase": 261,
    "title": "香港与巴黎的城市往来 · U80",
    "question": "A　香港与内地的往来越来越频繁。\nB　巴黎与里昂的往来越来越频繁。",
    "model": "Qwen3-4B",
    "snapshot": "6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad",
    "dims": {
      "layers": 36,
      "heads": 32,
      "width": 9728
    },
    "run": null,
    "caseId": null,
    "scope": "最小句对 · 观察",
    "objects": [
      {
        "kind": "MLP",
        "block": 30,
        "index": 80
      }
    ],
    "result": "逐句取 U80 在各 token 上最大的门控投影，再用同份语料的均值和标准差换算。A 减 B 为 +0.240 个标准差。这对句子只提供激活读数。",
    "metrics": [
      {
        "label": "A · 标准分",
        "value": "1.165"
      },
      {
        "label": "B · 标准分",
        "value": "0.925"
      },
      {
        "label": "A − B",
        "value": "+0.240"
      }
    ],
    "source": "research/neurons/goodfire/GF-L30-U80-local.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "S9.pairs[0]",
    "promptType": "原始最小句对",
    "series": "Claude",
    "artifactSha": "1174879499b5cd32915a7a3c3a8f4c0c0ad2cecc6cf1994b15992a527a5efac1",
    "snapshotReference": "research/mainline/H0/report_v5.json"
  },
  {
    "id": "p2983-confirm-000-01",
    "phase": 2983,
    "title": "Orion 档案 · 按印刷行标签找代码",
    "question": "In the active archive, find the CODE in the record with printed ROW D. Return the code alone, with no punctuation.",
    "model": "Qwen3-14B",
    "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
    "dims": {
      "layers": 40,
      "heads": 40,
      "width": 17408
    },
    "run": "clt_b26_three_cues_v3",
    "caseId": "confirm-000-01",
    "scope": "确认集 · 单题",
    "objects": [
      {
        "kind": "Head",
        "block": 26,
        "index": 36
      }
    ],
    "result": "模型答出 lion。B26/H36 在读到最后一个输入 token 时，对第 3 行的记录注意力最大，题目指定的是第 3 行。",
    "metrics": [
      {
        "label": "模型答案",
        "value": "lion"
      },
      {
        "label": "目标记录",
        "value": "第 3 行"
      },
      {
        "label": "峰值记录注意力",
        "value": "37.4%"
      }
    ],
    "source": "frontend/src/components/v8/threeCueEvidence.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "clt_b26_three_cues_v3 / confirm-000-01",
    "materialSource": "tests/glm5/result/clt_b26_three_cues_v3/confirm-materials.json",
    "artifactSha": "fd5db4808b38ffdfd5fcd25b023ee71f4bb624acd3871a94cd81d5723d3174ba",
    "promptType": "原题问句",
    "readout": "最后输入 token"
  },
  {
    "id": "claude261-u9517-s9-2",
    "phase": 261,
    "title": "两岸关系与健康饮食 · U9517",
    "question": "A　两岸关系是大陆和台湾共同关心的话题。\nB　健康饮食是大家共同关心的话题。",
    "model": "Qwen3-4B",
    "snapshot": "6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad",
    "dims": {
      "layers": 36,
      "heads": 32,
      "width": 9728
    },
    "run": null,
    "caseId": null,
    "scope": "最小句对 · 观察",
    "objects": [
      {
        "kind": "MLP",
        "block": 30,
        "index": 9517
      }
    ],
    "result": "逐句取 U9517 在各 token 上最大的门控投影，再用同份语料的均值和标准差换算。A 减 B 为 +4.564 个标准差。这对句子只提供激活读数。",
    "metrics": [
      {
        "label": "A · 标准分",
        "value": "5.377"
      },
      {
        "label": "B · 标准分",
        "value": "0.813"
      },
      {
        "label": "A − B",
        "value": "+4.564"
      }
    ],
    "source": "research/neurons/goodfire/GF-L30-U9517-local.json",
    "status": "本地历史摘录",
    "receipt": null,
    "sourceKey": "S9.pairs[1]",
    "promptType": "原始最小句对",
    "series": "Claude",
    "artifactSha": "878f2cb3b60428da80ac3160ef30a39e941bf138b025cae5cc59ef21a3a70ccf",
    "snapshotReference": "research/mainline/H0/report_v5.json"
  }
];
