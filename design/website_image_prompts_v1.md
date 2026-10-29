# 首页大图提示词 v1（DeepMind 视觉语言）

> 对象：`frontend/website/index.html` 的 4 张主图（hero 1 张 + 栏目卡 3 张，另有 1 张备用）
> 目标：从「2018 商业科技图库」迁到「2026 研究机构编辑部影像」——更简洁、更大气、更质感
> 状态：仅提示词与规格，站点文件未改动

---

## 0 现状诊断（为什么必须换）

| 位置 | 现有文件 | 内容 | 问题 |
|---|---|---|---|
| Hero | `index.png` 588×511 | 西装人群排排站，仰望发光地球，白色星点连线 | 图库四件套之一；人小、脸糊；放大到 670×540 后人群变成噪点。**注意：问题不在「地球」，而在「发光球 + 星点连线 + 仰望人群」这个组合**——地球本身可以留，换掉材质与拍法即可（见 §3） |
| 01 计划 | `assets/home_plan.png` 580×180 | 白色台阶通向云中城市 | 隐喻过直（阶梯=进步），塑料感强；台阶细节在 3.5:1 裁切下只剩一片白 |
| 02 AGI | `assets/home_agi.png` 599×175 | 发光线框大脑 + HUD 波浪 + 数据面板 | 最像「AI 素材包」的一张；深蓝底在整页白底里像一块补丁；HUD 面板暗示了并不存在的界面 |
| 03 社区 | `assets/home_forum.png` 590×174 | 会议室圆桌 + 巨幕世界地图 | 商务会议摄影，与「开放社区」气质相反；人物是图库脸 |
| 04 捐赠（已注释） | `assets/home_support.png` 599×162 | — | 备用，暂未上卡 |

共同缺陷：**发光 / 星点连线 / 半透明科幻感 / 图库人群 / HUD 面板** 这五样在 DeepMind 的官网上基本不出现。它们的共同问题是「用符号指代概念」，而 DeepMind 的做法是**拍一个具体的东西，让概念从物体本身长出来**。

需要说明一点：**地球本身不是禁忌**。DeepMind 自己也大量使用地球与大气影像（气候、天气、地球观测），区别在于他们拍的是「一颗被观测的行星」——哑光、有云、有光照方向、有科学仪器感；而不是「一颗发光的球加一圈连线」。所以 Hero 最终仍然以地球为主题，只换材质与拍法。

---

## 1 统一视觉语法（五张共用）

写进每张提示词的头部，保证四张摆在一页里像同一批拍摄的。

| 维度 | 规定 |
|---|---|
| **光** | 单一大面积柔光，来自左上；无硬高光、无轮廓光、无霓虹边。接触阴影长而柔，落进背景 |
| **材质** | 哑光陶瓷 / 磨砂玻璃 / 未涂布纸 / 拉丝铝 / 细丝。**禁止**：金属镜面、塑料亮面、发光体 |
| **色** | 骨白 `#f7f8fa`、暖灰 `#e6e8ec`、石墨 `#2a3038`、冰川蓝 `#7fb2d9`。全图最多一个冷调强调色，且**饱和度低于**站点主色 `#0284c7`——让页面的蓝是唯一的饱和色 |
| **构图** | 单一主体 + 60–70% 负空间；主体可悬浮，也可落在无缝浅色平面上；不做框式构图、不做暗角 |
| **相机** | 等效 50–100mm，f/2.8–f/8，略高于或略低于视平线的 3/4 视角；浅景深，前实后虚 |
| **质感** | 极细 35mm 颗粒；渐变平滑无断层；无彩边、无镜头光晕 |
| **禁止项**（每张都写） | 文字/字母/数字/水印/logo/UI/HUD/图表；发光连线网络；地球/大陆轮廓；电路板；二进制；机器人；人手；真实人脸；霓虹；镜头光晕；暗角；剪贴式投影 |

**三条贯穿的判断标准**
1. 缩到 240px 宽还能认出「一个东西」——靠轮廓和明暗，不靠细节。
2. 灰度打印出来仍然成立——如果去掉颜色就废了，说明它在靠色光骗人。
3. 能说出它是什么材质——说不出就是「渲染氛图」，不是质感。

---

## 2 CSS 约束（决定构图安全区，必读）

生成前必须按遮罩留白，否则好图也会被压掉一半。

| 图 | CSS 遮罩 | 语义 | 主体该放哪 |
|---|---|---|---|
| Hero `index.png` | `.hero-visual::before`：左侧白色 55% → 30% 处透明；`object-fit: contain`，右对齐 | 与左侧文案相接 | **右 55%**；左三分之一留空（会被洗白） |
| 01 `home_plan.png` | `.horizon`：白 0.92 → 40% 处 0.34 → 右侧 0 | 文左图右 | **右 55%**；左 40% 只放平缓地面 |
| 02 `home_agi.png` | `.brain`：左 0.06 暗 → 右 0.9 白 | 图左文右 | **左 55%**；右 30% 必须是干净浅底（会被洗成白） |
| 03 `home_forum.png` | `.forum`：白 0.92 → 40% 处 0.32 → 右侧 0 | 文左图右 | **右 55%**；左 40% 留平坦地面 |

**尺寸与比例**

| 用途 | 建议出图 | 理由 |
|---|---|---|
| Hero | `1500 × 1200`（5:4） | 渲染盒 670×540 = 1.24；源码 1.151 也在这个区间。`contain` 不裁切，比例必须接近 |
| 栏目卡 | `1400 × 400`（7:2） | 渲染盒 3.0–3.78 且 `cover`，7:2 上下各余 ~5% 可裁；不要低于 3:1，否则窄屏会被切掉主体上下 |
| 备用 04 | `1400 × 400` | 同上 |

---

## 3 Hero ·「石膏地球」= 人类共同体的尺度

**概念**：地球保留，但把它做成**石膏模型标本**——哑光、带工具痕、落在浅色平面上。同一颗星球，光球在喊，石膏球在陈述。
**两个决定质感的手感**：
1. **陆地凸起、海面抛光**：陆海只靠材质与阴影区分，不用颜色。这是整张图能不能「像真的」的分水岭。
2. **薄尘接光，而非辉光**：边缘只有一层悬浮的石膏粉在接光，所以它有极薄大气，但不发光。

**English prompt（主用）**

```
Editorial hero image for a scientific research institute, 3D still life render,
monochrome. A model globe carved from matte bone-white plaster, floating a few
centimetres above a seamless pale grey plane. The continents are raised in low relief
with crisp, true coastlines and gently sanded edges; the ocean surfaces are polished
smooth, so land and water separate purely by texture and by the shadow each ridge throws,
never by colour. No country borders, no labels, no grid, no latitude lines, no cloud.
A fine veil of plaster dust hangs at the surface, catching light only along the limb, so
the planet reads as having an extremely thin atmosphere rather than a glow.
Lighting: one large soft source from the upper left; a soft terminator falls across the
right side of the sphere, the unlit hemisphere sinking into cool grey without going
black; a long soft contact shadow on the plane below.
Palette: bone white, warm grey, cool slate; one note of glacier blue deep in the
recessed oceans. Camera: 85mm, f/4, three-quarter view just below centre, shallow depth
of field — the near limb crisp, the far limb softening.
Composition: the globe occupies the right two thirds, its left limb at roughly 38% of
frame width; the left third is empty pale ground.
Mood: a museum specimen, not a tech illustration — quiet, exact, planetary in scale,
restrained. Fine 35mm film grain, smooth gradients, no banding.
```

**中文提示词（即梦 / 可灵 / 混元可直接用）**

```
研究机构官网头图，单色 3D 静物渲染。一颗哑光骨白石膏雕成的地球模型，悬在无缝浅灰
平面上方几厘米。陆地是浅浮雕凸起，海岸线清晰准确、边缘有打磨过的柔角；海面被抛光
光滑，陆与海只靠材质和山脊投下的阴影区分，完全不用颜色。没有国界、没有标注、没有
经纬网格、没有云层。球面外浮着一层极细的石膏粉尘，只在球体边缘接住光，于是它有极
薄的大气感，而不是发光。左上方单一大面积柔光，柔和的晨昏线落在球体右侧，暗面沉入
冷灰但不发黑；下方平面上是细长的软接触阴影。配色：骨白、暖灰、冷石板灰，仅在凹陷
的海面深处有一点冰川蓝。85mm，f4，略低于中心的四分之三视角，浅景深，近端清晰、远端
柔化。地球位于画面右侧三分之二，球体左缘约在画面宽度 38% 处；左侧三分之一为空白浅
色地面。气质：博物馆标本，而非科技插画——安静、精确、行星尺度、克制。细颗粒，渐变
平滑无断层。
```

**负向词**

```
text, letters, numbers, labels, typography, watermark, logo, UI, HUD, country borders,
political map, latitude longitude grid, glowing, bloom, city lights, blue marble,
satellite photograph, clouds, weather system, network arcs, connection lines, satellites,
orbiting rings, circuit board, binary code, robot, human hands, crowd, neon, lens flare,
heavy vignette, black background, starfield, sparkles, plastic sheen, mirror chrome
```

**参数**：MJ `--ar 5:4 --style raw --stylize 120`；SD/Flux 建议 CFG 4–5，步数 30+

**这张唯一的风险**：地球题材最容易在缩到 240px 后变成一颗「白蛋」——大陆轮廓糊掉就什么都不剩。因此出图后必须先看 240px 缩略，确认至少**两块大陆的轮廓**仍然可辨（建议让非洲与欧亚同时入画，它们的形状辨识度最高），才进入下一步。

**备选 3B ·「单线地球」**（更抽象，也更「研究机构」）

一根不断的光丝缠绕成球，靠丝线疏密勾出大陆，其余留白：

```
A single unbroken filament of pale light wound into a sphere, thousands of turns, its
density mapping the continents — dense over land, sparse over ocean — so the planet is
drawn entirely by line weight and nothing else. Matte bone-white background, one soft
source from upper left, the far side of the sphere fading into haze. No glow except where
the winding is densest. 85mm, f/4, subject right of centre, left third empty. Mood:
cartography by hand, exact, quiet.
```

**备选 3C ·「高空云层」**（完全不画球体，最克制）

如果觉得「一颗球」还是太符号化，可以只给地球的一部分：从 40,000 英尺俯拍一片层积云，云顶在斜射光下形成柔和的高地与沟壑，海面在云隙间露出一点冷灰。观众自己会补全「这是地球」。`--ar 5:4`，主体在右，左三分之一留空。

---

## 4 01 ·「等高线地形 + 一条路」= 长期公共路线

**概念**：计划不是阶梯，是**地形**——一层层等高线是被时间压出来的高低，而那一条窄渠是路线。
**为什么替掉台阶**：台阶只能表达「上升」，表达不了「分叉、绕路、长期」；等高线天生带「地形需要勘测」的意思，这正是治理项目该有的语气。

```
Wide editorial banner, abstract 3D still life. A topographic model of gentle terraced
ridges, machined from matte bone-white plaster and stacked like contour maps — each
terrace edge is a clean, knife-thin ridge, every surface smooth and untextured. One
continuous narrow channel, a single engraved path, cuts a calm S-curve from the lower
left up across the terraces toward a low horizon; the channel is filled with a faint
pale-blue tone that is lit, never glowing. Lighting: raking soft light from the left at
a low angle, so the terraces read through shadow alone; no specular highlights.
Palette: bone white, cool grey, one thin line of desaturated sky blue. Camera: 60mm,
f/8, low three-quarter angle compressing the terraces into horizontal bands; left third
of the frame is empty pale ground. Mood: cartography, patience, governance, quiet
precision. Fine 35mm grain, clean gradients.
```

```
宽幅官网配图，抽象 3D 静物。哑光骨白石膏雕出的缓坡台地地形模型，像一层层叠起的
等高线地图；每层台地边缘是刀锋般干净的细棱，所有表面光滑无纹理。一条连续的窄渠
——唯一的一条刻痕路径——呈平静的 S 形，从左下穿过各层台地伸向低平的地平线；渠内
填入淡淡的浅蓝，是被照亮的，不是发光。左侧低角度掠射柔光，台地仅靠阴影显形，无
任何高光。配色：骨白、冷灰、一条低饱和天蓝细线。60mm，f8，低四分之三视角把台地压
成横向条带；画面左侧三分之一为空白浅色地面。气质：制图、耐心、治理、精确。细颗粒。
```

**负向词**：在统一负向词基础上追加 `staircase, steps, city skyline, clouds, glass, chrome, arrows, roadmap diagram, dotted line, bevel, glossy plastic`

**参数**：`--ar 7:2 --style raw --stylize 100`

---

## 5 02 ·「侧立的玻璃层 + 稀疏跨层细丝」= 语言机制解剖

**概念**：他们真正在做的事是**逐层切开模型看机制**——36 层、特征、跨层投影。用一叠侧立的磨砂玻璃片，加几根跨层的丝，把这件事拍出来。
**为什么替掉发光大脑**：大脑是「智能」的符号，不是「研究方法」的符号；而且深蓝底在整页白底里像补丁。玻璃层是**剖面**，天然对应「解剖」，也让这张图留在浅色调里。

```
Wide editorial banner, macro 3D render. A stack of thin, edge-on translucent glass plates
standing slightly ajar, like a rack of microscope slides or a sectioned block of layered
resin. Each plate is frosted and nearly clear, carrying an almost invisible engraved
microtexture — sparse fine dots and short filaments that read as data but never as
readable marks or letters. A few single hair-thin filaments of pale cold light bridge
between specific points on different plates, crossing the gaps at slight angles; they are
the only bright elements in the frame. Lighting: soft directional light from the left,
faint caustics pooling on a matte grey plane beneath; the right edge of the composition
dissolves into clean white haze. Palette: cool grey, clear glass, deep slate in the gaps,
one note of glacier blue. Camera: 90mm macro, f/2.8, very shallow depth of field — the
second and third plates are sharp, the rest drift out of focus. Mood: anatomy, dissection,
exactness; scientific, not science fiction.
```

```
宽幅官网配图，微距 3D 渲染。一叠侧立的薄玻璃片略微错开，像一排显微镜载玻片，或一块
被切成层的树脂。每片都是磨砂半透明，表面有几乎看不见的刻蚀微纹理——稀疏的细点和
短线，像数据但绝不构成可读字符。有几根发丝般细的极淡冷光丝在特定点之间跨层连接，
以微小的角度穿过间隙；它们是画面里唯一明亮的东西。左侧柔和方向光，下方哑光灰面上
有极淡的焦散；构图右缘化入干净的白雾。配色：冷灰、透明玻璃、间隙里的深石板灰，只
一点冰川蓝。90mm 微距，f2.8，极浅景深，第二三片实、其余虚。气质：解剖、精确，是科学
而非科幻。
```

**负向词**：追加 `brain, neurons, synapses, glowing brain, hud, dashboard, graphs, neon blue, dark navy background, sci-fi, hologram, wireframe mesh, binary, medical illustration, blood`

**参数**：`--ar 7:2 --style raw --stylize 100`

---

## 6 03 ·「一圈哑光球，各自有影子」= 开放社区

**概念**：社区是**一群彼此不同、各有位置的个体**，不是世界地图上的光点，也不是围坐的会议室。
**关键细节**：每个球有自己的影子，阴影在圈心叠成一潭——「各自独立」与「共同在场」同时成立。地面上只刻相邻连线，**不连到中心**，避免又长成网络图。

```
Wide editorial banner, elevated 3/4 overhead view of a physical still life. A loose open
ring of about forty small matte spheres of slightly varied diameter, resting on a seamless
pale grey plane. Each sphere is a different muted tone — bone, sand, pale slate, soft clay,
one or two dusty teal — all matte ceramic, no reflection, no glow, no gloss. Between some
neighbouring spheres, thin arcs are incised into the plane like pencil compass lines,
connecting only neighbours and never all to a single centre. Lighting: high, wide, very
soft top light; every sphere casts its own small soft shadow, the shadows overlapping into
one gentle dark pool inside the ring. Camera: 50mm, f/5.6, 35-degree elevation, modest
depth of field so the far spheres soften. Composition: the ring occupies the right two
thirds; the left third is empty pale plane. Mood: an assembly of individuals rather than a
network diagram — tactile, human-scaled, calm.
```

```
宽幅官网配图，俯视四分之三 3D 静物。约四十个大小略异的哑光小球，围成一个松散的开口
环，落在无缝浅灰平面上。每个球色调不同——骨白、沙、浅石板、柔陶土、一两个灰扑扑的
青——全部哑光陶瓷，无反射、无发光、无亮面。部分相邻球之间，平面上刻着像圆规画出的
细弧线，只连相邻，绝不全部汇向中心。顶部大面积柔和光，每个球落下自己的小软影，阴影
在环内叠成一潭淡淡的暗。50mm，f5.6，35 度俯角，适度景深，远端球体略虚。环占画面右
侧三分之二，左侧三分之一为空白浅色平面。气质：一群各自独立的个体，而不是网络图；
有触感、有人味尺度、平静。
```

**负向词**：追加 `world map, globe, meeting room, office, business people, suits, chairs, table, network diagram, hub and spoke, glowing dots, aerial city`

**参数**：`--ar 7:2 --style raw --stylize 110`

---

## 7 04 ·「棱镜分光」（备用，卡片当前已注释）

**概念**：透明与可核查——一束白光穿过棱镜，分成有节制的光谱。**不要**做成彩虹。
若恢复 04「捐赠与开支」卡再用。

```
A single solid glass prism standing on matte bone-white paper. A narrow beam of white
light enters from the left and separates into a restrained spectrum — muted, desaturated
bands of pale blue, mint, sand and dust rose — falling across the paper as a soft-edged
band of colour. No rainbow gaiety, no saturated primaries. Lighting: dark studio, one
narrow beam, everything else in soft shadow. Camera: 100mm macro, f/4, low angle, shallow
focus on the paper where the colour lands. Mood: transparency, audit, honest accounting.
```

---

## 8 落地清单

1. **出图**：按上表尺寸各出 1 张，先看 240px 缩略（三条判断标准），过不了重出，不要靠后期救。
2. **压片**：`cwebp -q 82 -resize 1400 0` → `home_plan.webp` 等；hero 出 1500 宽。现有 PNG 保留作回退。
3. **改引用**：`fusion-theme.css` 第 206/212/218 行三处 `url("assets/home_*.png")` 与 `index.html` 的 `src="index.png"` 换成 `.webp`。
4. **首屏重测**：`object-fit: contain` 的 hero 换比例后会变位置，需在 1440 / 1024 / 760 三档复测 `hero-visual img` 的 `getBoundingClientRect()`，确认主体没被左侧白遮罩吃掉。
5. **一致性复核**：四张并排截图，确认灰度下也是同一套明暗语言。

---

## 9 与旧版的一句话对比

旧图在说「我们有 AI」；新图在说「我们在做什么」。前者靠符号，后者靠物件——这正好是 DeepMind 官网与一般科技公司官网的分界。

Hero 保留了地球这个最容易被认出的主题，但把它从「发光的数据球」改成「一件石膏标本」：同一个对象，换一种拍法，观感就从宣传物料变成机构陈述。
