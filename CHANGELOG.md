# Changelog

本文件记录 ActionFlow 的重要变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- **段级评估工具**（`actionflow/evaluate.py`）
  - 输出 Precision / Recall / F1 / 类别准确率 / 平均 IoU / 边界误差
  - 段匹配用贪心最大 IoU，一个真值段不会被重复占用
  - `--oracle` 模式可验证匹配规则本身
- **学习指南**（`docs/learning-guide.html`）
  - 8 大算法分别对应到具体类与行号
  - 四天学习路线、每个算法的验证命令、调参对照表
- **CI**：GitHub Actions 上跑 lint + 124 个测试 + 端到端 smoke test
- Issue 模板：Bug 报告 / 功能建议

### 修复

- **`--arch boundary` / `--arch fusion` 无法导出 ONNX**
  - boundary：包装类未继承底座模型的 `num_joints`，导出器造 dummy 输入时崩溃
  - fusion：内部注意力掩码过不了新版 dynamo 导出器，改走 `dynamo=False`
  - 两者均已补回归测试（`tests/test_model.py::test_all_archs_export_onnx`）
- **训练保存的配置不可跨机器使用**
  - 配置里存的是绝对路径，换台机器就找不到姿态权重
  - 现在写盘时存相对项目根目录的路径，读取时自动还原
- **`scripts/*.py` 直接运行 ImportError**
  - 脚本目录不是项目根目录，`import actionflow` 失败；现在自动把根目录插进 `sys.path`
- **`add_rest_class.py` 不幂等**
  - 重复运行会重复补「静止」段；现在检测到已含静止段就跳过
- `docs/README.md` 里指向不存在的 `architecture.md` 的死链

### 变更

- 删除未被引用的 `actionflow/utils/geometry.py`（功能已被 `augment_skeleton` 覆盖）
- 删除 `pipeline.Timer`（与 `utils.logging.Stopwatch` 重复）
- 删除 `ntu25_to_coco17_xy`（无引用）
- `technical-plan.html` 移除与面试准备相关的章节，章节号重排（20 → 18 节）

## [0.1.0] — 2025-11

首个公开版本。

### 新增

- **Layer 1 · 姿态提取**
  - RTMPose ONNX 推理封装（含 SimCC 亚像素坐标解码）
  - One-Euro 自适应滤波
  - 身体坐标系归一化（位置 / 尺度 / 朝向不变）
- **Layer 2 · 动作分类**
  - 骨架时序 Transformer（`offline`）
  - 因果流式分类器（`streaming`）—— 前缀不变性实测差异 0.0
  - 边界回归分类器（`boundary`）—— 时间戳精确到帧
  - 视频路 Cross-Attention 双路融合（`fusion`）
- **Layer 3 · 时序解码**
  - Viterbi 动态规划、Hysteresis 双阈值、最小持续时长约束
  - 边界回归修正（中位数稳健估计消解边界歧义）
- **数据工具链**
  - 自动抽关键点、GUI 标注（tkinter）、命令行标注、标注质量校验
  - 训练窗口导出、标签表生成、按受试者划分
- **部署**
  - ONNX 导出（流式架构专用导出器）+ 导出后自动推理验证
  - 离线推理 / 摄像头实时 / 边播边出结果三种运行模式
- 124 个单元测试
