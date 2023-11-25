# 贡献指南

感谢你愿意花时间改进 ActionFlow。

这份文档只讲**具体怎么做**，不重复 README 里已有的内容。

---

## 开发环境

```bash
git clone <your-fork-url>
cd actionflow

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` 装的是 pytest + ruff + mypy。日常开发不需要 `[all]`（那个含 torch，跑起来慢）。

---

## 提交前必跑

```bash
ruff check actionflow/ tests/ tools/ examples/ scripts/   # lint，必须 0 error
python -m pytest tests/ -q                                 # 124 个测试，必须全绿
bash scripts/demo.sh                                       # 端到端自检
```

三条都过了再开 PR。CI 跑的也是这三条。

> `scripts/demo.sh` **不需要**任何权重、不需要数据集、不需要网络。
> 如果它挂了，说明你改坏了主干链路 —— 这条底线不能破。

---

## 代码风格

由 `ruff` 统一管理，配置在 `pyproject.toml`：

| 规则 | 说明 |
|---|---|
| 行宽 100 | `E501` 已忽略（长字符串和 URL 很难折行） |
| 规则集 | `E` `F` `W` `I` `N` `UP` `B` |
| `I`（import 排序） | **最容易忘**。跑 `ruff check --fix` 自动修 |
| `N803` `N806` | 允许大写变量名 —— `T` / `B` / `J` / `C` 是张量维度，不是变量名违规 |

---

## 写代码时最重要的一条约定

> **在 docstring 里写清楚「为什么这么设计」，而不只是「做了什么」。**

这个项目里最有价值的注释都是这个形态。比如：

```python
# ❌ 不够好
def normalize_keypoints(kp):
    """归一化关键点。"""

# ✅ 好
def normalize_keypoints(kp):
    """
    把关键点转到身体坐标系。

    ⚠️ 为什么要做:
        人在画面左边还是右边、身高 160 还是 190,
        都不应该影响「抬左手」这个判断。
        不做归一化的话,同一个动作在不同位置会被当成不同特征,
        训练集必须覆盖各种位置才学得会 —— 数据量直接爆炸。
    """
```

**为什么这条这么重要**：这个仓库的价值在于每个决策背后的理由。
只写「做了什么」的代码，半年后没人敢改。

---

## 新功能要做什么

1. **写测试** —— 参考 `tests/` 里的写法。
   测试是**纯脚本**（不依赖 pytest 也能跑），每个文件末尾有 `_run_all()`：
   ```python
   def _run_all() -> int:
       tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
       ...
   ```
   这样 CI 里不会因为 pytest 插件问题挂掉。

2. **更新文档**
   - 算法相关 → `docs/learning-guide.html`（在哪个类、怎么验证）
   - 架构相关 → `docs/technical-plan.html`（为什么这么设计）
   - 用法相关 → `README.md`

3. **更新 CHANGELOG** —— 在 `[Unreleased]` 下加一条。

---

## 提交信息

用中文或英文都行，说清楚**做了什么**和**为什么**：

```
fix: 修复 --arch boundary 模型无法导出 ONNX

包装类 BoundaryAwareClassifier 没有继承底座模型的 num_joints,
导出器构造 dummy 输入时抛 AttributeError。
已补回归测试覆盖四种架构。
```

---

## 目录结构约定

```
actionflow/
├── core/         配置与常量 —— 改这里等于改全局行为
├── io/           输入输出 —— 视频源、姿态推理、可视化
├── models/       Layer 2 —— 四种架构
├── postprocess/  Layer 1 后处理 + Layer 3 解码
├── data/         数据集与标注工具
├── infer/        端到端管线（胶水层）
├── utils/        日志与计时
├── cli.py        命令行入口
├── train.py      训练
├── export.py     ONNX 导出
└── evaluate.py   段级评估
```

**加新东西时先想清楚它属于哪一层。** 跨层调用是这里最需要避免的事。

---

## 数据集相关的约定

- **数据集不进仓库**。`.gitignore` 里 `/data/` 整个目录被排除。
- 想分享数据集请单独发 Release，并写清楚自己的 License。
- NTU RGB+D 是**非商业研究用途**，不要打包进仓库。

---

## 许可

提交的代码按 [Apache-2.0](LICENSE) 授权。
如果你提交了第三方代码或模型权重，请在 PR 里说明来源和许可。
