# ActionFlow 文档索引

| 文档 | 内容 |
|---|---|
| [`learning-guide.html`](learning-guide.html) | ★**学习指南** —— 8 大算法分别在哪个类、哪一行，怎么验证 |
| [`technical-plan.html`](technical-plan.html) | **完整技术方案**（18 节，含算法对比表与选型分析） |
| [../README.md](../README.md) | 项目总览、快速开始、**数据制作全流程** |

> **两个 HTML 文档怎么配合看**
>
> | 你想解决的问题 | 看哪个 |
> |---|---|
> | 「这个算法在哪个文件？我要去读它」 | **learning-guide** |
> | 「为什么这么设计？有没有别的做法？」 | **technical-plan** |
> | 「数据怎么录？命令怎么敲？」 | **README** |

---

## learning-guide.html 章节速查

| # | 章节 | 什么时候看 |
|---|---|---|
| 01 | 先搞清楚：项目在干什么 | 第一次接触，先建地图 |
| 02 | **算法总览表（8 个）** | ★打印出来贴显示器上 |
| 03 | 算法 1-3：输入侧三个手写件 | One-Euro / 归一化 / 关节嵌入 |
| 04 | 算法 4-5：模型侧两个核心件 | ★★**因果掩码**在这里，最重要 |
| 05 | 算法 6-8：输出侧三个手写件 | Viterbi / Hysteresis / 最小时长 |
| 06 | 附加件：边界回归（可选架构） | 时间戳不精确时看 |
| 07 | 附加件：双路融合（可选架构） | 骨架分不清语义时看 |
| 08 | **学习路线：四天读完** | ★照着这个顺序读 |
| 09 | 怎么验证你真的懂了 | 每个算法的验证命令 |
| 10 | 常见困惑 FAQ | 卡住的时候查 |

---

## technical-plan.html 章节速查

| # | 章节 | 什么时候看 |
|---|---|---|
| 01 | 心智模型：这个项目就三步 | 第一次接触 |
| 02 | RTMPose 姿态估计 | 装权重、调推理 |
| 03 | Transformer 动作分类 | **写模型前必读** |
| 04 | 时序解码（三个手写算法） | **写解码前必读** |
| 05 | 完整架构与数据流 | 理解整体 |
| 06 | 训练：数据集与训练循环 | 训模型 |
| 07 | 工程结构与三条命令 | 跑起来 |
| 08 | 开发路线图 | 排期 |
| 09 | 开源协议避坑 | 发开源前 |
| 10 | 下一步 | 规划 |
| 11 | **三层架构详解** | 理解「姿态+时序」为什么不够 |
| 12 | **算法全景与对比分析** | 理解选型依据 |
| 13 | **三个手写模块 + 为什么有解码层** | **看不懂算法时看这个** |
| 14 | 流式因果架构**方案设计** | 做实时版本 |
| 15 | **因果注意力掩码（已实现）** | 看流式怎么落地 + 踩的坑 |
| 16 | **边界回归头（已实现）** | 看时间戳怎么做到帧级精度 |
| 17 | **双路融合 + Cross-Attention（已实现）** | 看骨架路怎么补语义 |
| 18 | **自建数据集与标注工具（已实现）** | 看数据从哪来 |

---

## 核心结论速记

**项目 = 三层，不是两层**
```
① 姿态识别 → 每帧骨架（几何事实）
② 动作分类 → 每帧类别（语义标签）  ← 常被漏掉
③ 时序解码 → 动作段（时空结构）
```

**八个算法分别在哪**
```
算法 0  RTMPose + SimCC    io/pose_infer.py         外部库封装
算法 1  One-Euro Filter    postprocess/filters.py    手写 ★
算法 2  身体坐标系归一化   postprocess/normalize.py  手写 ★
算法 3  关节嵌入+位置编码   models/classifier.py      标准用法
算法 4  因果注意力掩码     models/streaming.py       手写 ★★核心
算法 5  Key-Padding Mask   models/streaming.py       标准用法
算法 6  Viterbi 动态规划   postprocess/decode.py     手写 ★
算法 7  Hysteresis 双阈值  postprocess/decode.py     手写 ★
算法 8  最小持续时长约束   postprocess/decode.py     手写 ★
```

**三个手写模块的分工**
```
One-Euro    → 治「输入脏」（关键点抖动）    分类之前
Hysteresis  → 治「段被切断」（边界抖动）    解码时
Viterbi     → 治「局部最优」（孤立误判）    解码时
```

**四种模型架构**
```
offline   → 整段视频打一个标签（baseline）
streaming → 逐帧输出,因果掩码,前缀不变性实测 0.0   ★默认
boundary  → 流式 + 边界回归头,时间戳精确到帧
fusion    → 骨架路 + 视频路 Cross-Attention（可选增强）
```

**你会用 / 不会用**
```
✅ RTMPose、MMPose、nn.TransformerEncoder、One-Euro、Hysteresis、Viterbi
✅ 因果注意力掩码、边界回归头、Cross-Attention、Softplus
⚠️ 知道原理但先不用：ST-GCN、BSN、CRF、因果 TCN
—  不用：整个视频路线（VideoMAE/SlowFast/TimeSformer）
```
