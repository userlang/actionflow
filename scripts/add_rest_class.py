"""给现有标注自动补「静止」段:动作段的补集(开头/间隙/结尾)。

用法:
    python scripts/add_rest_class.py

原标注先备份到 data/annotations_bak/。

为什么需要「静止」类
------------------
模型只学「做了什么动作」的话,视频里大量的静止时间(人站着不动、
动作之间换姿势的间隙)没有对应的标签。训练时这些帧会带着噪声标签进来,
把「没有动作」误判成某个动作。

补上「静止」段之后:
    ① 每一帧都有明确标签,没有「未知」的空隙
    ② 模型学会区分「在动」和「没在动」
    ③ 实时推理时不会在休息间隙误报一个动作

⚠️ 补出来的「静止」段**不是**逐帧核对的真值 —— 它是「除了标注的
动作以外的所有时间」的补集。动作段和静止段的边界可能差几帧,
这是可接受的(标注工具本身也建议动作前后留静止)。

⚠️ 幂等:已经含「静止」段的标注文件会被跳过,不会重复补。
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REST = "静止"
MIN_GAP_FRAMES = 20  # <0.7s 的碎缝不值得标,跳过

ANN_DIR = ROOT / "data" / "annotations"
BAK_DIR = ROOT / "data" / "annotations_bak"

if not ANN_DIR.exists():
    print(f"找不到标注目录: {ANN_DIR}")
    raise SystemExit(1)

BAK_DIR.mkdir(parents=True, exist_ok=True)

for jp in sorted(ANN_DIR.glob("*.json")):
    data = json.loads(jp.read_text(encoding="utf-8"))
    v = data["videos"][0]
    acts = sorted(v["actions"], key=lambda a: a["start_frame"])

    # ── 幂等检查:已经补过就跳过 ────────────────────────────────
    if any(a["label"] == REST for a in acts):
        print(f"{jp.name}: 已含「{REST}」段,跳过")
        continue

    new_acts = []
    prev_end = 0
    for a in acts:
        if a["start_frame"] - prev_end >= MIN_GAP_FRAMES:
            new_acts.append({"label": REST, "start_frame": prev_end,
                             "end_frame": a["start_frame"]})
        new_acts.append(a)
        prev_end = a["end_frame"]
    if v["num_frames"] - prev_end >= MIN_GAP_FRAMES:
        new_acts.append({"label": REST, "start_frame": prev_end,
                         "end_frame": v["num_frames"]})

    if REST not in data["labels"]:
        data["labels"].append(REST)

    n_rest = sum(1 for a in new_acts if a["label"] == REST)
    if n_rest == 0:
        print(f"{jp.name}: 没有可补的空隙,跳过")
        continue

    backup = BAK_DIR / jp.name
    if not backup.exists():
        shutil.copy2(jp, backup)
    v["actions"] = new_acts
    jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{jp.name}: +{n_rest} 段静止 → 共 {len(new_acts)} 段")
