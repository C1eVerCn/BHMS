#!/usr/bin/env python3
"""兼容入口：准备 NASA 训练数据。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    cmd = [sys.executable, str(Path(__file__).with_name('prepare_datasets.py')), '--source', 'nasa', '--include-in-training']
    raise SystemExit(subprocess.call(cmd))
