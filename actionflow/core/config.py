"""全局配置。所有可调参数集中在此，避免散落在代码各处。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

# 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"


@dataclass
class PoseConfig:
    """Layer 1 · 姿态估计配置"""

    onnx_path: str = str(WEIGHTS_DIR / "rtmpose-m.onnx")
    input_size: tuple[int, int] = (192, 192)  # (width, height)
    input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    input_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # RTMPose / MMPose 的 SimCC 输出分桶数
    simcc_split_ratio: float = 2.0
    # 使用哪个推理后端: "cuda" | "cpu"
    providers: list[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    # 低于该置信度的关键点视为漏检
    keypoint_thresh: float = 0.3
    # 输入为 BGR 还是 RGB
    color_order: str = "RGB"


@dataclass
class FilterConfig:
    """One-Euro Filter 配置

    参数选择的经验法则:
      - `min_cutoff` 越小 → 平滑越强 → 抖动压得越狠，但延迟越大。
        视频帧率 30fps 时，1.0 是保守起点，0.3~0.5 更激进。
      - `beta` 是速度耦合系数 —— 越大，快速运动时越"敢少滤波"。

        ⚠️ 重要：检测噪声本身会产生很大的表观速度，beta 偏大会把噪声
        误判成"快速运动"，从而**削弱滤波**。实测在 σ=3px 的检测噪声下:
            beta=0.02 → 抖动只降 19%
            beta=0.00 → 抖动降 37%
        因此默认取 0，只在确实关心"挥手时不能有拖影"时再调大。
    """

    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0


@dataclass
class NormalizeConfig:
    """身体坐标系归一化配置"""

    enabled: bool = True
    # 躯干尺度参考点：(肩中点, 髋中点)
    torso_scale: str = "torso"  # "torso" | "bone_median"
    # 是否旋转到标准朝向（+y 朝下、头顶在负 y，与图像坐标系一致）
    rotate_to_canonical: bool = True
    # 最小有效躯干长度（像素），低于此值认为检测失败，跳过归一化
    min_torso_length: float = 1.0


@dataclass
class ModelConfig:
    """Layer 2 · 动作分类器配置"""

    num_classes: int = 60
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    window: int = 32  # 窗口长度 T（帧）
    num_joints: int = 17
    # 关键点置信度阈值：低于此值视为漏检，该关节不参与注意力
    conf_threshold: float = 0.3
    # ── 模型类型 ────────────────────────────────────────────────
    # "offline" : 整段池化，回答"这段视频整体是什么动作"
    # "streaming": 因果流式，逐帧输出（满足前缀不变性）
    #
    # 详见 actionflow/models/streaming.py
    arch: str = "streaming"
    # 因果注意力掩码（仅 arch="offline" 且 causal=True 时启用）
    # ⚠️ 注意:普通分类器的 causal 掩码是"帧内全连通 + 帧间因果",
    #    与流式模型一致。若只是为了流式推理,请用 arch="streaming"。
    causal: bool = False
    # 是否启用边界回归头（提升时间戳精度，见 actionflow/models/boundary.py）
    use_boundary_head: bool = False


@dataclass
class DecodeConfig:
    """Layer 3 · 时序解码配置"""

    # Hysteresis 双阈值
    high_threshold: float = 0.7
    low_threshold: float = 0.4
    # 最小持续时长（秒），低于此长度的段丢弃
    min_duration: float = 0.3
    # 是否启用 Viterbi 平滑
    use_viterbi: bool = True
    # Viterbi 转移矩阵参数
    stay_prob: float = 0.9
    switch_prob: float = 0.05
    # 相邻同类段是否合并（间隔小于 merge_gap 秒）
    merge_gap: float = 0.0
    # 边界回归最多能修正多少帧（防止回归头给出离谱值时边界跑飞）
    max_boundary_adjust: int = 10


@dataclass
class ActionFlowConfig:
    """ActionFlow 总配置"""

    pose: PoseConfig = field(default_factory=PoseConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    norm: NormalizeConfig = field(default_factory=NormalizeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    # 类别名称表（留空则用 "action_0" 形式）
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionFlowConfig:
        cfg = cls()
        for key, value in d.items():
            if not hasattr(cfg, key):
                continue
            sub = getattr(cfg, key)
            # ⚠️ 判断"是不是 dataclass"要用 is_dataclass()。
            #    isinstance(sub, dataclass) 是错的 —— dataclass 是装饰器
            #    不是类型,这条路径(从 checkpoint 的 config.json 恢复配置)
            #    一执行就 TypeError。测试没覆盖到加载配置这条路,漏网了。
            if is_dataclass(sub) and isinstance(value, dict):
                for k2, v2 in value.items():
                    if hasattr(sub, k2):
                        setattr(sub, k2, v2)
            else:
                setattr(cfg, key, value)
        return cfg


def _relocate_weights(cfg: ActionFlowConfig) -> ActionFlowConfig:
    """
    把配置里的姿态权重路径解析成本机可用的**绝对路径**。

    为什么需要这一步
    ----------------
    ``train.py`` 每次训练都会在权重旁存一份 ``<权重>.config.json``,
    里面记着当时生效的 pose.onnx_path。而 ``PoseConfig.onnx_path``
    的默认值是 ``str(WEIGHTS_DIR / "rtmpose-m.onnx")`` —— 一个
    **绝对路径**。

    于是这份配置一旦离开生成它的机器就彻底失效:

        "onnx_path": "/Users/jacklang/project/actionflow/weights/rtmpose-m.onnx"

    别人 clone 下来用同一个权重推理,会直接报
    ``FileNotFoundError: 找不到 ONNX 权重`` —— 而那个文件明明就在
    ``weights/rtmpose-m.onnx``。

    三种情况的处理
    --------------
    ① **相对路径**(写盘时存的就是这种)→ 按项目根目录还原成绝对路径。
       不还原的话,只有在项目根目录下运行才能找到权重。
    ② **绝对路径且本机存在** → 原样保留(用户可能指定了项目外的权重)。
    ③ **绝对路径但本机不存在** → 尝试按「相对项目根目录」重新解释。
       典型的场景:权重在别的机器上训练过,配置里存的是对方的绝对路径。

    这样配置就能跟着权重一起在机器之间流通。
    """
    p = Path(cfg.pose.onnx_path)

    # ① 相对路径 → 按项目根目录还原
    if not p.is_absolute():
        cfg.pose.onnx_path = str(PROJECT_ROOT / p)
        return cfg

    # ② 本机存在 → 原样保留
    if p.exists():
        return cfg

    # ③ 本机不存在 → 尝试重新解释成「相对项目根目录」
    try:
        rel = p.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = None

    if rel is None:
        # 路径前缀整个对不上(比如在 /Users/alice/... 下训练,
        # 现在跑在 /home/bob/... 下)。这时退一步:
        # 取 "weights/" 之后的部分,按本机的项目根目录重新拼。
        parts = p.parts
        if "weights" in parts:
            idx = len(parts) - 1 - parts[::-1].index("weights")
            rel = Path(*parts[idx:])
        else:
            # 不在本项目目录下(用户自定义的权重路径)—— 原样保留,
            # 强行改写反而会让权重找不回来。
            return cfg
    cfg.pose.onnx_path = str(PROJECT_ROOT / rel)
    return cfg


def _portable_dict(cfg: ActionFlowConfig) -> dict[str, Any]:
    """
    生成可移植的配置字典 —— 绝对路径换成相对项目根目录的路径。

    与 ``_relocate_weights`` 的区别:这个函数**不修改**传入的 cfg,
    只返回一份用于写盘的副本。训练流程里 cfg 还要继续用
    (推理时要靠它找姿态权重),不能被写盘动作顺带改掉。
    """
    import copy

    d = copy.deepcopy(cfg.to_dict())
    p = Path(str(d.get("pose", {}).get("onnx_path", "")))
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            pass  # 项目外的自定义权重,原样保留
        else:
            d["pose"]["onnx_path"] = str(rel)
    return d


def load_config(path: str | Path) -> ActionFlowConfig:
    """从 YAML/JSON 加载配置。"""
    import json

    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("读取 YAML 需要安装 pyyaml：pip install pyyaml") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    return _relocate_weights(ActionFlowConfig.from_dict(raw))


def save_config(cfg: ActionFlowConfig, path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 存相对路径,让配置能跟着权重跨机器流通(见 _portable_dict)
    path.write_text(
        json.dumps(_portable_dict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
