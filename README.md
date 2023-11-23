<div align="center">

# ActionFlow

**流式人体原子动作识别引擎** — 视频 → 带时间戳的结构化动作序列

`输入一段真人视频（或摄像头实时流） → 在线输出 [(0.0-1.2s) 左抬腿, (1.2-2.5s) 右手握拳, ...]`

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-112%2F112%20passing-brightgreen.svg)](tests/)

</div>

---

## 这是什么

一个端侧可跑的**人体原子动作识别**引擎。它不是「给整段视频打一个标签」，而是：

- **逐帧**判断每一帧在做什么动作
- **精确到帧**地切分出动作段，输出起止时间戳
- 支持**流式**处理（摄像头实时流），边跑边出结果
- 模型仅 **0.32M 参数**，可导出 ONNX 在手机 / 树莓派 / 嵌入式运行

```
视频流
  ↓
┌─────────────────────────────────────────────────────┐
│ Layer 1  POSE ESTIMATION  姿态识别                   │
│   RTMPose (ONNX)  →  17 关键点                       │
│   One-Euro Filter  →  去抖                           │
│   身体坐标系归一化  →  位置/尺度/朝向不变              │
└─────────────────────────────────────────────────────┘
                    ↓ 骨架流 [T, 17, 3]
┌─────────────────────────────────────────────────────┐
│ Layer 2  ACTION RECOGNITION  动作分类  ★核心          │
│   关节 Embedding + 时间位置编码                       │
│   nn.TransformerEncoder (2层, d=128)                  │
│   因果注意力掩码 (arch=streaming,真正的流式)          │
│   Key-Padding Mask (处理漏检)                        │
│   边界回归头 (arch=boundary,精确到帧)                │
└─────────────────────────────────────────────────────┘
                    ↓ 逐帧概率 [T, C] + 边界距离 [T, 2]
┌─────────────────────────────────────────────────────┐
│ Layer 3  TEMPORAL LOCALIZATION  时间定位               │
│   Viterbi 动态规划  →  全局最优路径                   │
│   Hysteresis 双阈值 →  抗边界抖动                     │
│   最小持续时长约束  →  滤除假阳性                     │
│   边界回归修正  →  把迟到的边界拉回来                 │
└─────────────────────────────────────────────────────┘
                    ↓
  [{"start":0.0, "end":1.2, "label":"左抬腿", "confidence":0.94}, ...]
```

---

## 四种模型架构

ActionFlow 支持四种架构，用 `arch` 字段切换。它们共用同一套骨架编码器，
区别在**注意力掩码**和**额外的头**：

| `arch` | 类 | 输出 | 参数量 | 用途 |
|---|---|---|---|---|
| `offline` | `SkeletonTransformer` | `(B, C)` | 0.114M | 整段视频打一个标签（baseline） |
| `streaming` | `CausalStreamingClassifier` | `(B, T, C)` | 0.110M | ★默认。真正的流式：逐帧输出，历史结果永不改写 |
| `boundary` | `BoundaryAwareClassifier` | `(B, T, C)` + `(B, T, 2)` | 0.114M | 流式 + 精确到帧的时间戳 |
| `fusion` | `FusionClassifier` | `(B, T, C)` | 0.354M | 骨架路 + 视频路双路融合 |

```bash
# 训练流式模型
python -m actionflow.train --data data/ --arch streaming --epochs 30

# 训练带边界回归的模型（时间戳更准）
python -m actionflow.train --data data/ --arch boundary --epochs 30
```

### 1. 因果注意力掩码（真正的流式架构）

普通 Transformer 是「上帝视角」——第 3 帧能看到第 8 帧。
这在离线视频里没问题，但在**实时流里是致命的**：
用户已经看到的第 3 帧的结果，会因为你后来又来了 5 帧而**偷偷改变**。

本项目的做法是给注意力加一层**因果掩码**（causal mask），
让每个 token 只能看「不晚于自己」的 token。

> **通俗解释**：注意力机制就像开会，每个 token 都要听所有人发言。
> 因果掩码就是规定「只能听前面的人的发言，不能听后面的」——
> 这样第 3 帧的结论一旦算出来就**永远不会变**。

有个坑值得记下来：掩码必须**按帧号**算，不能按 token 号算。

```
token 编号 = 帧号 × 关节数 + 关节编号

按 token 编号做下三角掩码 → 同一帧内的 17 个关节互相看不见 ❌
  （左手腕看不到左手肩 —— 这在物理上说不通）

按帧号做掩码 → 同帧内 17 个关节互相可见,帧与帧之间才因果 ✅
```

**前缀不变性**是流式架构的验收标准，代码里有自动测试：

```
「用前 t 帧算出的结果」必须和「用前 t+100 帧算出的前 t 帧」**完全一样**
实测差异:streaming = 0.0,boundary < 1e-5,fusion < 1e-5
```

### 2. 边界回归头（提升时间戳精度）

Hysteresis 解码靠「置信度跨过阈值」判定边界，天然**偏晚**：
动作刚开始时模型还没把握（置信度只有 0.5），要等涨过 0.7 才敢开段。

边界回归头换了个问法：

| | 问题 | 性质 |
|---|---|---|
| 分类头 | 「这一帧是**什么**动作？」 | 离散、粗糙 |
| 边界回归头 | 「这一帧距**最近的边界**还有几帧？」 | 连续、精细 |

它输出两个非负距离（用 `softplus` 保证）：

```
dist_to_start[t] = 这一帧距「它所属那一段」的起点还有几帧
dist_to_end[t]   = 这一帧距「它所属那一段」的终点还有几帧
```

于是边界位置可以直接反算出来：

```
起点 = t - dist_to_start[t]
终点 = t + dist_to_end[t] + 1        （终点是开区间）
```

> **通俗解释**：分类头像是在问「这段视频是讲什么的」，
> 边界回归头像是在问「我还需要几帧才到下一个镜头切换」。
> 后者对时间点的刻画精细得多。

**修正算法**（`postprocess/decode.py::refine_boundaries_with_regression`）：

```
第 1 遍:对段内帧做中位数统计,估计这一段真正的起点/终点
第 2 遍:解决相邻段的冲突（边界是「共享资源」,谁挪了就大家都挪）
第 3 遍:重算置信度
```

> ⚠️ 为什么不能直接读段首帧？因为**段首帧是歧义帧** ——
> 它可能是新段的第 1 帧，也可能是上一段的最后 1 帧，
> 两种含义在数值上分不出来。只有段**内部**的帧才不含歧义。
>
> 通俗讲：不信「门口那一帧」，看「屋里 10 个人都怎么说」，
> 取多数意见（中位数）。一两个异常值带不偏结果。

### 3. 视频路 backbone + Cross-Attention 融合

骨架路看到的是**几何**，看不到**语义**。举几个例子：

- 「拿起杯子」和「拿起手机」—— 手的位置一样，但手里有东西 vs 手里是空的，骨架完全一样
- 「开门」和「挥手」—— 手的轨迹可能相似，但门把手的有无不同
- 「走路」和「跑步」—— 关节角度接近，但步频、步幅不同

视频路直接看像素，颜色、纹理、物体全在里面。融合方式用 **Cross-Attention**：

```
骨架路问视频路：「这一帧画面上有什么？」 → 得到 语义补充
视频路问骨架路：「这个人怎么动的？」     → 得到 几何引导
```

> **通俗解释**：Cross-Attention 就像「查字典」。
> 骨架特征是查询词，视频特征是字典，
> 每个骨架 token 问「和我最相关的画面信息是什么」，
> 注意力机制算出相关度，加权求和。

关键设计是**残差连接**：`融合 = 骨架原样 + 视频补充`。
即使视频路认错了（画面里有干扰物），骨架信息还在，
模型顶多「参考了一下错误意见」，不会彻底跑偏。

`VideoBackbone` 是抽象接口，接什么都能用：

```python
from torchvision.models.video import swin3d_t
net = swin3d_t(weights="DEFAULT")   # 换成任意预训练视频 backbone
```

本仓库自带 `SimpleVideoBackbone`（3× Conv3d）用于跑通链路和测试，
**不用于刷 SOTA**。默认配置仍是骨架路（0.11M 参数，端侧可跑）。

### 4. 自建数据集与标注工具

公开数据集（NTU RGB+D / Kinetics）有三个问题：

1. **粒度不对** —— NTU 是「打电话 / 吃饭 / 读书」这种复合动作，我们要的是「抬左手 / 右手握拳」这种原子动作
2. **没有边界标注** —— 我们需要「第几帧到第几帧在做什么」
3. **场景不匹配** —— NTU 是实验室固定机位，真实场景各种角度各种光照

所以必须自建。工具支持**关键点辅助标注**：
先用 RTMPose 自动抽关键点，人只做少量修正（拖动标错的点、标类别和起止帧）。
标一段 5 秒视频，从 30 分钟降到 5 分钟。

```bash
# ① 从视频自动抽关键点
python -m actionflow.data.annotate extract --video clips/s01_b.mp4 --out data/a.npz

# ② 起标注界面（tkinter，macOS 自带）
python -m actionflow.data.annotate gui --video clips/s01_b.mp4 --npz data/a.npz \
    --out annotations/a.json

# ③ 校验标注质量（重叠/越界/类别不均衡）
python -m actionflow.data.annotate validate --annotations annotations/ --npz-dir data/

# ④ 导出成训练用的 npz 切片
python -m actionflow.data.annotate export --annotations annotations/ --npz-dir data/ \
    --out data/windows/
```

GUI 快捷键：`空格` 播放/暂停 · `←/→` 逐帧 · `A/D` 加/减关键点 · `Enter` 标记段起点 · `N` 下一段 · `Delete` 删除当前段 · `Ctrl+S` 保存 · `Ctrl+Q` 退出


---

## 快速开始

### 安装

```bash
git clone https://github.com/your-org/actionflow.git
cd actionflow

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### 三条命令跑通

```bash
# ① 看看解码层在干什么（无需任何依赖，纯算法演示）
python -m actionflow.cli decode-test

# ② 端到端流水线自检（无需 ONNX 权重 / 无需数据集）
python tools/verify_pipeline.py

# ③ 生成一段合成测试视频（无需下载数据集）
python tools/make_demo_video.py --out assets/demo.mp4

# ④ 四种架构演示（流式 / 边界回归 / 双路融合，全部用随机骨架）
python examples/architectures.py
```

或者一条命令跑完全部检查：

```bash
bash scripts/demo.sh          # 生成视频 → 自检 → 解码演示 → 112 个单元测试
```

### 免下载跑通姿态提取

仓库里带了一个 **28KB 的 mock RTMPose 权重**（`weights/mock-pose.onnx`），
所以不需要去 MMPose 下几百 MB 的模型就能验证整条链路：

```bash
python -m actionflow.cli pose \
    --video assets/demo.mp4 \
    --onnx weights/mock-pose.onnx \
    --visualize output/vis.mp4
```

### 完整推理

```bash
# 提取姿态并可视化
python -m actionflow.cli pose --video demo.mp4 --visualize out.mp4 --out poses.npz

# 端到端推理
python -m actionflow.cli infer --video demo.mp4 --model weights/actionflow_skel.pth --out result.json

# 实时摄像头
python -m actionflow.cli live --model weights/actionflow_skel.pth
```

### 训练

```bash
python -m actionflow.train --data /path/to/nturgbd --epochs 30 --batch-size 64
python -m actionflow.export --model weights/actionflow_skel.pth --out weights/af.onnx
```

---

## 项目结构

```
actionflow/
├── actionflow/
│   ├── core/                    # 核心配置与常量
│   │   ├── config.py            #   全部可调参数集中管理
│   │   └── constants.py         #   COCO17 关键点定义、骨架连边
│   ├── io/                      # 输入输出层
│   │   ├── pose_infer.py        #   RTMPose ONNX 推理封装
│   │   ├── video.py             #   文件/摄像头/目录 三种输入源统一接口
│   │   └── visualize.py         #   骨架绘制、动作时间轴条
│   ├── models/                  # Layer 2 模型
│   │   ├── classifier.py        #   骨架时序 Transformer 分类器（离线版）
│   │   ├── streaming.py         #   ★因果流式分类器 + 因果掩码构造
│   │   ├── boundary.py          #   ★边界回归头 + 边界损失 + 标签构造
│   │   ├── fusion.py            #   ★视频路 backbone + Cross-Attention 融合
│   │   └── factory.py           #   模型工厂（按 arch 统一构建）
│   ├── postprocess/             # 后处理 + Layer 3 解码
│   │   ├── filters.py           #   One-Euro Filter
│   │   ├── normalize.py         #   身体坐标系归一化
│   │   └── decode.py            #   Hysteresis + Viterbi + 边界回归修正
│   ├── data/                    # 数据集
│   │   ├── ntu.py               #   NTU RGB+D 解析（骨架直读）
│   │   ├── dataset.py           #   Dataset + 骨架增强 + 逐帧标签
│   │   ├── annotate.py          #   ★标注数据结构 + 校验 + 导出
│   │   ├── annotate_cli.py      #   ★标注工具命令行入口
│   │   └── annotate_gui.py      #   ★标注 GUI（tkinter）
│   ├── infer/
│   │   └── pipeline.py          #   端到端管线（离线/流式两种模式）
│   ├── utils/                   # 日志、几何工具
│   ├── cli.py                   # 命令行入口
│   ├── train.py                 # 训练脚本
│   ├── export.py                # ONNX 导出
│   └── configs/default.yaml     # 默认配置
├── tools/
│   ├── make_demo_video.py       # 生成合成测试视频
│   ├── mock_pose_onnx.py        # 生成 mock RTMPose 权重（免下载跑通全流程）
│   └── verify_pipeline.py       # 流水线自检
├── tests/                       # 112 个单元测试（7 个文件）
│   ├── test_decode.py           #   9 个：解码
│   ├── test_postprocess.py      #   15 个：滤波 + 归一化
│   ├── test_model.py            #   11 个：模型 + ONNX 导出
│   ├── test_streaming.py        #   19 个：★因果掩码 + 前缀不变性
│   ├── test_boundary.py         #   25 个：★边界回归 + 边界修正
│   ├── test_fusion.py           #   15 个：★Cross-Attention + 融合
│   └── test_annotate.py         #   18 个：★标注工具
├── examples/
│   ├── quickstart.py            #   各层 API 用法示例
│   └── architectures.py         #   ★四种架构演示（流式/边界/融合）
├── docs/
│   ├── README.md                # 文档索引
│   └── technical-plan.html      # 完整技术方案（20 节）
└── weights/                     # 模型权重（mock-pose.onnx 已随仓库提供）
```

---

## 设计要点

### 为什么三层而不是一层

| 层 | 给出什么 | 学术名称 |
|---|---|---|
| ① 姿态识别 | 手在哪 | Human Pose Estimation |
| ② 动作分类 | 这是抬手 | Action Recognition |
| ③ 时序解码 | 何时起止 | Temporal Action Localization |

**②③ 缺一不可**。只有 ①+③ 你能把视频切成段，但不知道每段是什么动作。
只有 ①② 你能知道每帧是什么，但输出的是碎帧，没有时间戳。

### 为什么要有解码层

逐帧独立分类会产生三类错误：

1. **段被劈碎** — 中间一帧误判，一个连续动作被切成两段
2. **孤立假阳性** — 某帧置信度突增，输出一个 0.03s 的不可能动作
3. **边界抖动** — 同一动作两次推理起止帧差 3~5 帧

解码层用三种手段各治一种：

```
Viterbi 动态规划  → 治「孤立误判」（全局最优路径）
Hysteresis 双阈值 → 治「段被切断」（迟滞回环，同 Canny 边缘检测）
最小时长约束      → 治「假阳性段」（人不会做 0.03 秒的动作）
```

> 多数开源动作识别项目只做到 Layer 2 就停了，直接把 argmax 当输出，边界全是碎的。

### 为什么 One-Euro 的 beta 默认是 0

检测噪声本身会产生很大的表观速度，`beta` 偏大会把噪声误判成「快速运动」，
反而**削弱**滤波。实测 σ=3px 噪声下：

| beta | 抖动降低 |
|---|---|
| 0.02 | 19% |
| 0.00 | **37%** |

只有在确实关心「挥手时不能有拖影」时才调大 `beta`。

### 为什么流式模型要做「前缀不变性」测试

因为**流式架构的错误在离线测试里是看不出来的**。

假设模型不小心让第 3 帧看到了第 8 帧：
- 离线推理：把整段视频喂进去，结果看着挺好，测试全绿
- 实时推理：用户已经看到「抬手」，下一秒突然变成「站立」—— 因为新帧到了

所以必须有一个**独立的验收标准**：

```
「前 t 帧算出的结果」 == 「前 t+100 帧算出的前 t 帧」
```

这不是「差不多就行」，是**逐位相同**。这条测试写在 `tests/test_streaming.py` 里，
任何改动破坏了它都会立刻挂掉。实测差异：`streaming = 0.0`（逐位相同）。

> **通俗解释**：这就像直播 —— 弹幕可以后到，但你说出的话不能被撤回。
> 前缀不变性就是「说过的话不反悔」的数学保证。

### 为什么边界修正要用中位数

因为**边界帧是歧义帧**。

假设真值是「抬手=[30,70)，站立=[70,120)」，但 Hysteresis 迟到了，
把边界切在 69。第 69 帧被分给了「站立」，可真值里它属于「抬手」。

现在看第 69 帧的回归预测：
- 按「站立」的语义读 → 距站立起点 0 帧 → 边界就在 69 ✓
- 按「抬手」的语义读 → 距抬手起点 39 帧 → 边界在 30 ✓

**两种读法在数值上都合法**，光看这一帧根本分不出来。
但段**内部**的每一帧都指向同一个答案 —— 抬手段内的第 40 帧说
「距起点 10 帧」→ 起点 = 40 − 10 = 30，句句都指向 30。

所以做法是：**不信门口那一帧，看屋里的人怎么说的，取多数意见（中位数）**。

---

## 相关技术

| 层 | 用到的算法 | 出处 |
|---|---|---|
| 姿态 | **RTMPose** (SimCC 坐标分类编码) | MMPose, Apache-2.0 |
| 姿态 | One-Euro Filter | Casiez et al. 2012, CHI |
| 时序 | **nn.TransformerEncoder** + 关节/位置嵌入 | Vaswani et al. 2017 |
| 时序 | 因果注意力掩码（流式架构） | Vaswani et al. 2017（decoder mask） |
| 时序 | Key-Padding Mask | — |
| 时间定位 | **边界回归头**（软标签 + L1 损失） | ActionFormer (ICCV'21) |
| 时间定位 | Softplus（保正回归） | — |
| 融合 | **Cross-Attention** 双路融合 | Vaswani et al. 2018 |
| 解码 | **Viterbi** (隐马尔可夫解码) | Rabiner 1989 |
| 解码 | Hysteresis 双阈值 (迟滞回环) | 同 Canny 边缘检测, 1986 |
| 解码 | 中位数稳健估计（边界歧义消解） | — |
| 部署 | **ONNX Runtime** | Microsoft, MIT |

完整算法对比与选型分析见 [`docs/technical-plan.html`](docs/technical-plan.html)。

---

## 训练数据

| 数据集 | 用途 | 许可 |
|---|---|---|
| [NTU RGB+D 60](https://github.com/lshiwjx/nturgbd) | 骨架动作 60 类，主力 baseline | ⚠️ 非商业研究用途，**不要打包进仓库** |
| [Kinetics](https://github.com/google-deepmind/kinetics-i3d) | 视频预训练 | ⚠️ Google 条款 |
| 自建 | 30~60 个原子动作 × 多人 × 多视角 | 自己声明 |

**自建数据集**是本项目的护城河 —— 公开数据集没有「原子动作」粒度。

---

## 开发

```bash
pip install -e ".[dev]"

ruff check actionflow/ tests/ tools/ examples/   # lint（0 error）
ruff check --fix actionflow/                     # 自动修

python tests/test_decode.py       # 9 个：解码
python tests/test_postprocess.py  # 15 个：滤波 + 归一化
python tests/test_model.py        # 11 个：模型 + ONNX 导出
python tests/test_streaming.py    # 19 个：因果掩码 + 前缀不变性
python tests/test_boundary.py     # 25 个：边界回归 + 边界修正
python tests/test_fusion.py       # 15 个：Cross-Attention + 融合
python tests/test_annotate.py     # 18 个：标注工具
```

测试是**纯脚本**写的（不依赖 pytest），每个文件末尾有 `_run_all()`，
所以 `python tests/test_xxx.py` 直接就能跑，CI 里也不会因为 pytest 插件问题挂掉。

代码风格约定：
- 行宽 100，Ruff 默认规则集（E/F/W/I/N/UP/B）
- `T` / `B` / `J` / `C` 保留大写 —— 它们是张量维度，不是变量名违规

---

## License

代码 [Apache-2.0](LICENSE)。模型权重若来自 MMPose 预训练（Apache-2.0）可随仓库分发；
**数据集不打包进仓库**，请自行下载并遵守各自许可。

---

## 路线图

- [x] Layer 1 姿态提取 + One-Euro 滤波 + 坐标归一化
- [x] Layer 2 骨架时序 Transformer 分类器
- [x] Layer 3 Viterbi + Hysteresis 时序解码
- [x] 离线 / 流式双模式管线
- [x] ONNX 导出
- [x] **边界回归头**（借鉴 ActionFormer，提升时间戳精度）
- [x] **因果注意力掩码**（真正的流式架构，前缀不变性实测 0.0）
- [x] **视频路 backbone（Video Swin-T 接口）+ Cross-Attention 融合**
- [x] **自建原子动作数据集与标注工具**（CLI + GUI + 质量校验）

### 接下来

- [ ] 训一个真实权重（当前只有 mock 姿态模型）
- [ ] 端侧 ONNX 部署实测（Android / 树莓派）
- [ ] 双路融合的量化压缩
- [ ] Web 演示页（把结果可视化成人形动画）
