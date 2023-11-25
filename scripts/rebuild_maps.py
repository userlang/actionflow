"""重建 labels.json + splits（按人划分）。

用法:
    python scripts/rebuild_maps.py

作用:
    ① 扫描 data/windows/ 下所有 npz,按「样本数降序」建立稳定的
       标签 → id 映射,写到 data/labels.json
    ② 按**人**(视频名里下划线前的前缀,如 s01)划分训练/测试集,
       写到 data/splits/train.txt 和 test.txt

    ⚠️ 每次重新导出窗口后都要跑一次这个脚本 ——
       标签 id 会变,旧的权重就对不上了。

⚠️ 为什么需要手动加 sys.path:
    直接 `python scripts/rebuild_maps.py` 时,Python 把 **脚本所在目录**
    (scripts/) 放进 sys.path[0],而不是项目根目录,于是
    `import actionflow` 会 ModuleNotFoundError。
    下面这三行把项目根目录插到最前面,让脚本在任何工作目录下都能跑。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from actionflow.data.annotated import build_label_map, build_split  # noqa: E402

WINDOWS = ROOT / "data" / "windows"

if not WINDOWS.exists():
    print(f"找不到窗口目录: {WINDOWS}")
    print("请先执行: python -m actionflow.data.annotate_cli export \\")
    print("        --annotations data/annotations --npz-dir data/npz --out data/windows")
    raise SystemExit(1)

print(build_label_map(WINDOWS, out_json=ROOT / "data" / "labels.json"))
print(build_split(WINDOWS, ROOT / "data" / "splits"))
