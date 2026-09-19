

## Phase 2804（自动续研，LPF-17：k≥3 多义词条带普查——P-C7 补救 turned 理论修正：实体感定律）

**测试原理**：2803 的 P-C7 FALSE 归因于"k=2 时 dW₁=−dW₂ 反对称、PR 比按构造无定义"，当时唯一可判定词是 apple（ratio 4.8）。2804 冻结 20 词 × 3 感词表（turkey/shell/squash/mint/gold/silver/bronze/salmon/olive/bow/drum/port/bench/trunk/boot/horn/diamond/club/tank/polish），感域来自 14 个 2803 锚集 + 7 个新域（country/anatomy/money/weapon/container/card/geometry，全部 tokenizer 预检单 token）。协议沿 2803（Wmeans_s→dW_s→CM=E/rms·g·dW_s→ρ/SVD/PR/default/画像），**新增去自参照协议**：若 target∈自身感锚集则移除该锚词再算 Wmeans（2803 存在同型缺陷：apple∈fruit、iron∈metal、olive∈fruit、table∈furniture）。gates：G1 E-vs-atlas 0.000000；G2 含自参照 apple CM 精确复现 2802（0.00e+00）；G3 自参照效应量化 max|ΔCM|=0.0587（微小，2803 结论稳健）；G4 iron pr_ratio 复现 2803（err 0.025）。预注册 P-A8/P-B8/P-C8/P-D8/P-E8 于任何 embedding 读出前冻结（execution 3a6c730f）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-A8 画像可读 | TRUE | 60/60=100% 命中（turkey country 读出 america/brazil/china/france…，shell company 读出 google/microsoft/amazon/ibm/intel） |
| P-B8 对比编码 | TRUE | 20/20 ρ 矩阵全负 |
| P-C8 PR 分层 | **FALSE** | 仅 3/20（turkey 5.1/shell 3.6/polish 2.9）——k≥3 判据有定义后仍不普适，**apple 是例外非规律，P-C7 的 FALSE 升级为经验否定** |
| P-D8 命名<自然 | FALSE | 12/26=46%——"命名感紧凑"不是定律 |
| P-E8 平坦谱 | TRUE | 20/20（0.63-0.98） |

**头条发现（实体感定律）**：3 个分层词恰是 20 词中唯一带国家/公司实体感的词，且实体感 PR 全部取该词最小值（turkey country 32<84/162；shell company 90<244/326；polish country 47<132/134）——3/3。回看 apple：pr=[fruit 250, company 66, plant 318, food 226]，min=company 66，ratio 4.8 同样由 company 感驱动——**含实体指称感（country/company）的词 4/4 全部分层，无实体感词 17/17 全部平坦（ratio≤1.9），21/21 完美分离**。2803 的"命名感紧凑"修正为"实体指称感紧凑"：指称唯一实体（Apple 公司、Poland、Turkey 国家）的感锚定在稀疏维度（PR 32-90），指称开放类成员（颜色/运动/音乐风格等人为范畴）的感与自然类一样分布式（PR 130-360）。机制证据：实体感画像直接命中锚集实体词本身（turkey country→国名列表），范畴感读出锚词+属性词混合。

**辅助结果**：Arm C 连续性 4/4——turkey/salmon/olive/trunk 在去自参照+新增感后默认感 winner 全部稳定（food/animal/fruit/plant）；2803 自参照缺陷效应仅 0.059（apple CM 尺度），既有登记结论无需修正。

**相关文件**（SHA256 前 16）：脚本 phase2804_rdc_polysemy_k3.py=93521ee60fad5e4a；产物 phase2804/qwen4_polysemy_k3/{execution 3a6c730ff942504a, result f91ba18d74bf5657（20 词全记录+26 配对表）, k3.npz e0c670fef81feb87}。数据源：2795 atlas（2c0010ce）+ 2802 polysemy（ef6729e2，G2 对拍）+ 2803 result（iron G4 对拍）。tokenizer 预检拦截：joker/spade/kiwi/blackberry 2-tokens 全弃换（card 锚集补 chip/suit）；一次性跑通零调试。

**问题硬伤**：① 实体感定律 n=4（apple 复用 2802）统计弱，需大样验证；② 20 词全部 k=3，k=4 词表（date/cell 类）未覆盖；③ named/natural 二分粗糙（color/music/sport/vehicle 等人为范畴的语义地位需更细分类）；④ 画像命中判定为子串匹配，弱判据。

**结论**：多义谱三普适定律定稿——对比编码（ρ 全负）、等强对比轴（平坦谱）、画像可读性在 k≥3 词条带上全部满格复现；PR 分层被证伪为普适属性并重构为实体感定律（实体指称感=唯一系统性紧凑感类）。词嵌入的参数作用=它在所有语义对比方向上的带符号投票表（2802 终态定义）不受影响；新增分层维度：投票谱的宽度由感的指称类型决定。

**接续（2805 候选）**：(a) 实体感定律大样验证（30+ 实体感词 vs 30+ 无实体感词，预注册 21/21 分离的可重复性）；(b) 实体感紧凑机制（实体词 W_U 行几何：锚集实体词行间 cos vs 范畴词行间 cos——实体词族是否在 W_U 空间扎堆）；(c) 默认感排序伪困惑度行为效度（对接 2796 遗留）；(d) k=4 词表补全。
