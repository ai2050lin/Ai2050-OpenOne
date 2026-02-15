
import os
import sys

import numpy as np
import requests
import torch

# 模拟法律核心知识库
LEGAL_FIBER_DATA = [
    {
        "key": "contract_law_basic",
        "fact": "合同法基本原则：平等、自愿、公平、诚信、守法与公序良俗。",
        "logic_coordinates": [0.5, 0.8, -0.2], # 映射到逻辑流形的特定位置 (假设为“民法”区域)
    },
    {
        "key": "criminal_liability_age",
        "fact": "刑事责任年龄：已满16周岁的人犯罪，应当负刑事责任。",
        "logic_coordinates": [-0.7, 0.1, 0.9], # 映射到“刑法”区域
    },
    {
        "key": "property_rights",
        "fact": "物权法：不动产物权的设立、变更、转让和消灭，经依法登记，发生效力。",
        "logic_coordinates": [0.3, -0.4, 0.6], # 映射到“物权”区域
    }
]

def inject_knowledge():
    print("--- 启动 Phase VIII: 知识注入实验 ---")
    API_URL = "http://localhost:5001/nfb/fiber/register"
    
    for item in LEGAL_FIBER_DATA:
        print(f"正在注入纤维: {item['key']}...")
        # 模拟生成语义内容向量 (实际应由 Encoder 生成)
        content_vec = np.random.randn(128).tolist() 
        
        payload = {
            "key": item["key"],
            "content": content_vec,
            "pos": item["logic_coordinates"]
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"成功挂载到几何坐标: {item['logic_coordinates']}")
            else:
                print(f"注入失败! 状态码: {response.status_code}, 响应内容: {response.text}")
        except Exception as e:
            print(f"连接服务器失败: {e}")

    print("\n[实验结论]: 法律专业纤维库已成功在逻辑流形上完成几何挂载。")

if __name__ == "__main__":
    inject_knowledge()
