# -*- coding: utf-8 -*-
"""M2-2 patch: 摘除叙事入口（INPUT_PANEL_TABS SNN/ICSPB）、清理死状态语义、补登概念画廊名录。
每处锚点 assert 唯一命中；改后回读复核。"""
import io

REPORT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_m22_patch.txt"
out = []

def patch(path, old, new, tag, count=1):
    with io.open(path, 'r', encoding='utf-8') as f:
        t = f.read()
    n = t.count(old)
    assert n == count, "%s anchor hit %d times (expected %d)" % (tag, n, count)
    t2 = t.replace(old, new)
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(t2)
    out.append("%s OK (hit=%d)" % (tag, n))

# --- 1) config/panels.js: INPUT_PANEL_TABS 收敛为 DNN 主工作台 ---
P_PANELS = r"D:\AI2050\Ai2050-OpenOne\frontend\src\config\panels.js"
OLD_TABS = """// 输入面板标签
export const INPUT_PANEL_TABS = [
  { id: 'main', label: 'DNN', color: '#38bdf8', description: 'DNN 主工作台' },
  { id: 'snn', label: 'SNN', color: '#4ecdc4', description: '脉冲神经网络' },
  { id: 'icspb', label: 'ICSPB', color: '#6c5ce7', description: '当前模型工作台' },
];"""
NEW_TABS = """// 输入面板标签（M2 收敛：旧叙事面板 SNN/ICSPB 已摘除入概念画廊 /gallery，
// 组件本体保留待 M3 拆分处理；主导航只留 DNN 主工作台）
export const INPUT_PANEL_TABS = [
  { id: 'main', label: 'DNN', color: '#38bdf8', description: 'DNN 主工作台' },
];"""
patch(P_PANELS, OLD_TABS, NEW_TABS, "PANELS_INPUT_TABS")

# --- 2) App.jsx: 死状态 activeTab 默认值从 glass_matrix 改为 logit_lens ---
P_APP = r"D:\AI2050\Ai2050-OpenOne\frontend\src\App.jsx"
OLD_STATE = "  const [activeTab, setActiveTab] = useState('glass_matrix');"
NEW_STATE = "  const [activeTab, setActiveTab] = useState('logit_lens'); // M2: 死状态语义清理（setter 无调用点），不再指向概念件"
patch(P_APP, OLD_STATE, NEW_STATE, "APP_DEAD_STATE")

# --- 3) ConceptGallery.jsx: 名录补登本轮摘除的入口 ---
P_GAL = r"D:\AI2050\Ai2050-OpenOne\frontend\src\components\app\ConceptGallery.jsx"
OLD_SEC4 = """    items: ['App.jsx: snn_system / icspb_system / main_workspace / main_system tab', 'AGICentralCommand', 'HLAIBlueprint', 'GlobalTopologyDashboard'],"""
NEW_SEC4 = """    items: ['App.jsx: snn_system / icspb_system / main_workspace / main_system tab', 'AGICentralCommand', 'HLAIBlueprint', 'GlobalTopologyDashboard', '输入面板 SNN / ICSPB 旧叙事页（FiberNetPanel / SNNResearchDashboard，M2 起主导航摘除）', '结构面板死分支 agi / glass_matrix / flow_tubes / global_topology（M2 起无导航入口，M3 拆分时清理）'],"""
patch(P_GAL, OLD_SEC4, NEW_SEC4, "GALLERY_SEC4")

# --- 回读复核 ---
import os
for p, key, tag in [
    (P_PANELS, "旧叙事面板 SNN/ICSPB 已摘除", "PANELS_VERIFY"),
    (P_APP, "死状态语义清理", "APP_VERIFY"),
    (P_GAL, "M2 起主导航摘除", "GALLERY_VERIFY"),
]:
    t = io.open(p, encoding='utf-8').read()
    assert key in t, "%s failed readback" % tag
    out.append("%s: readback OK" % tag)

io.open(REPORT, 'w', encoding='utf-8').write("\n".join(out))
print("M2_2_PATCH_OK")
