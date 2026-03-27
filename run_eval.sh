#!/usr/bin/env bash
set -euo pipefail

# ========= 用户配置 =========
REPO_DIR="$HOME/le-wm"
VENV_DIR="$REPO_DIR/.venv"
STABLEWM_HOME="/mnt/data/szeluresearch/stable-wm"

# ========= 参数 =========
MODEL="${1:-dinowm}"
TASK="${2:-tworoom}"

# ========= 环境初始化 =========
echo "=== Activate environment ==="
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

echo "Python: $(which python)"
echo "Pip: $(which pip)"

# ========= 设置数据路径 =========
export STABLEWM_HOME="$STABLEWM_HOME"
echo "STABLEWM_HOME=$STABLEWM_HOME"

# 如果只有一张卡，固定到 GPU 0
export CUDA_VISIBLE_DEVICES=0

# ========= 基本检查 =========
echo "=== Check dataset ==="
if [ ! -f "$STABLEWM_HOME/${TASK}.h5" ]; then
    echo "❌ Missing dataset: $STABLEWM_HOME/${TASK}.h5"
    exit 1
fi

echo "=== Check checkpoint ==="
if [ ! -f "$STABLEWM_HOME/${TASK}/${MODEL}_object.ckpt" ]; then
    echo "❌ Missing checkpoint: $STABLEWM_HOME/${TASK}/${MODEL}_object.ckpt"
    exit 1
fi

# ========= 运行 =========
echo "=== Running eval ==="
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Policy: $TASK/$MODEL"
echo ""

python eval.py --config-name="${TASK}.yaml" policy="${TASK}/${MODEL}"