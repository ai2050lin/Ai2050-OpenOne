// Offline excerpts from Phase 2983 confirmation materials and Phase 2933 native MLP activations.
window.modelExplorerData = {
  "run": "clt_b26_three_cues_v3",
  "snapshot": "1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce",
  "cases": [
    {
      "id": "confirm-000-00",
      "rule": "按姓名",
      "rows": [
        {
          "name": "Kifota",
          "label": "C",
          "value": "silver",
          "mass": 0.13418025
        },
        {
          "name": "Tebaka",
          "label": "E",
          "value": "wood",
          "mass": 0.40520933
        },
        {
          "name": "Tumade",
          "label": "D",
          "value": "lion",
          "mass": 0.04257776
        },
        {
          "name": "Loneru",
          "label": "A",
          "value": "glass",
          "mass": 0.04523783
        },
        {
          "name": "Sigabu",
          "label": "B",
          "value": "fire",
          "mass": 0.10341178
        }
      ],
      "gold": 1,
      "output": "wood",
      "prompt": "<|im_start|>user\nThis archive contains five record lines. A printed ROW label is a letter and may appear in any order. For physical position use the order of the record lines as displayed. The old note is only background. The table is bounded by BEGIN and END lines. Read the CODE on the specified data line. Answer the final question with a single lowercase CODE word.\nOld note: NAME Tebaka; printed ROW D; physical position first.\nActive Archive Orion:\nBEGIN\nROW C | NAME Kifota | CODE silver\nROW E | NAME Tebaka | CODE wood\nROW D | NAME Tumade | CODE lion\nROW A | NAME Loneru | CODE glass\nROW B | NAME Sigabu | CODE fire\nEND\nQuestion: In the active archive, find the CODE in the record named Tebaka. Return the code alone, with no punctuation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:\n"
    },
    {
      "id": "confirm-000-01",
      "rule": "按印刷标签",
      "rows": [
        {
          "name": "Kifota",
          "label": "C",
          "value": "silver",
          "mass": 0.05200485
        },
        {
          "name": "Tebaka",
          "label": "E",
          "value": "wood",
          "mass": 0.08311543
        },
        {
          "name": "Tumade",
          "label": "D",
          "value": "lion",
          "mass": 0.37414223
        },
        {
          "name": "Loneru",
          "label": "A",
          "value": "glass",
          "mass": 0.03470675
        },
        {
          "name": "Sigabu",
          "label": "B",
          "value": "fire",
          "mass": 0.06246334
        }
      ],
      "gold": 2,
      "output": "lion",
      "prompt": "<|im_start|>user\nThis archive contains five record lines. A printed ROW label is a letter and may appear in any order. For physical position use the order of the record lines as displayed. The old note is only background. The table is bounded by BEGIN and END lines. Read the CODE on the specified data line. Answer the final question with a single lowercase CODE word.\nOld note: NAME Tebaka; printed ROW D; physical position first.\nActive Archive Orion:\nBEGIN\nROW C | NAME Kifota | CODE silver\nROW E | NAME Tebaka | CODE wood\nROW D | NAME Tumade | CODE lion\nROW A | NAME Loneru | CODE glass\nROW B | NAME Sigabu | CODE fire\nEND\nQuestion: In the active archive, find the CODE in the record with printed ROW D. Return the code alone, with no punctuation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:\n"
    },
    {
      "id": "confirm-000-02",
      "rule": "按物理首行",
      "rows": [
        {
          "name": "Kifota",
          "label": "C",
          "value": "silver",
          "mass": 0.10375173
        },
        {
          "name": "Tebaka",
          "label": "E",
          "value": "wood",
          "mass": 0.09131672
        },
        {
          "name": "Tumade",
          "label": "D",
          "value": "lion",
          "mass": 0.05560481
        },
        {
          "name": "Loneru",
          "label": "A",
          "value": "glass",
          "mass": 0.34138101
        },
        {
          "name": "Sigabu",
          "label": "B",
          "value": "fire",
          "mass": 0.03861831
        }
      ],
      "gold": 0,
      "output": "silver",
      "prompt": "<|im_start|>user\nThis archive contains five record lines. A printed ROW label is a letter and may appear in any order. For physical position use the order of the record lines as displayed. The old note is only background. The table is bounded by BEGIN and END lines. Read the CODE on the specified data line. Answer the final question with a single lowercase CODE word.\nOld note: NAME Tebaka; printed ROW D; physical position first.\nActive Archive Orion:\nBEGIN\nROW C | NAME Kifota | CODE silver\nROW E | NAME Tebaka | CODE wood\nROW D | NAME Tumade | CODE lion\nROW A | NAME Loneru | CODE glass\nROW B | NAME Sigabu | CODE fire\nEND\nQuestion: In the active archive, find the CODE on the data line just below BEGIN. Return the code alone, with no punctuation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:\n"
    }
  ],
  "feature": {
    "run": "B36-TRANSITION14-20261114-v1",
    "caseId": "t01eb4d20ccb2964:counted0",
    "unit": 2429,
    "block": 36,
    "positions": [
      {
        "name": "输入末尾",
        "index": 188,
        "value": 2.109375
      },
      {
        "name": "最后一步前",
        "index": 194,
        "value": -0.4765625
      },
      {
        "name": "Value 前",
        "index": 200,
        "value": -0.447265625
      },
      {
        "name": "Code 前",
        "index": 210,
        "value": 0.79296875
      }
    ],
    "groupSize": 32,
    "baseCorrect": 90,
    "blockedCorrect": 90,
    "total": 92
  }
};
