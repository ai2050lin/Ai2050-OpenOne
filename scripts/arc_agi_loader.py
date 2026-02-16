import json
import os
import random

import torch


class ARCLoader:
    """
    ARC-AGI 任务解析器，负责加载并格式化几何推理任务。
    """
    def __init__(self, data_path="data/arc-agi"):
        self.data_path = data_path
        self.tasks = []
        if os.path.exists(data_path):
            self.load_all_tasks()
        else:
            print(f"[!] Warning: ARC-AGI 数据路径 {data_path} 不存在。请下载数据集。")

    def load_all_tasks(self):
        # 遍历数据集文件夹
        for split in ['training', 'evaluation']:
            path = os.path.join(self.data_path, split)
            if not os.path.exists(path): continue
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    with open(os.path.join(path, filename), 'r') as f:
                        self.tasks.append(json.load(f))
        print(f"[+] Loaded {len(self.tasks)} ARC-AGI tasks.")

    def get_random_task(self):
        if not self.tasks: return None
        return random.choice(self.tasks)

    def format_task_as_token_stream(self, task):
        """
        将 ARC 的 Grid 格式转化为 LLM 可理解的序列（用于 FiberNet 语义扫描）。
        """
        train_examples = task['train']
        stream = "ARC Task Geometry:\n"
        for i, ex in enumerate(train_examples):
            stream += f"Example {i} Input: {ex['input']} Output: {ex['output']}\n"
        return stream

if __name__ == "__main__":
    # 模拟运行
    loader = ARCLoader()
    task = loader.get_random_task()
    if task:
        print(loader.format_task_as_token_stream(task))
