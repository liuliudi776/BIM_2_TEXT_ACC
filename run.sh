#!/bin/bash

# 生成唯一的会话名称
SESSION_NAME="multi_run_$(date +%s)"

# 创建新的tmux会话，并激活bimacc环境
tmux new-session -d -s "$SESSION_NAME"

# 配置文件列表
CONFIG_FILES=(
    # "config/bridge_config-Qwen2.5-72B-Instruct.toml"
    "config/bridge_config-Qwen2.5-32B-Instruct.toml"
    "config/bridge_config-Qwen2.5-14B-Instruct.toml"
    "config/bridge_config-Qwen2.5-7B-Instruct.toml"
    "config/bridge_config-4o-mini.toml"
    "config/bridge_config-deepseek-v3.toml"
    # # 建筑
    # # "config/building_config-Qwen2.5-72B-Instruct.toml"
    "config/building_config-Qwen2.5-32B-Instruct.toml"
    "config/building_config-Qwen2.5-14B-Instruct.toml"
    "config/building_config-Qwen2.5-7B-Instruct.toml"
    "config/building_config-4o-mini.toml"
    "config/building_config-deepseek-v3.toml"
)

# 为每个配置文件创建窗格并运行命令
for i in "${!CONFIG_FILES[@]}"; do
    if [ $i -gt 0 ]; then
        # 从第二个配置开始，创建新窗格
        tmux split-window -t "$SESSION_NAME"
        tmux select-layout -t "$SESSION_NAME" tiled
    fi
    
    # 在窗格中激活 conda 环境并运行程序
    tmux send-keys -t "$SESSION_NAME" "conda activate bimacc && python main.py --config ${CONFIG_FILES[$i]}" Enter
done

# 附加到tmux会话
tmux attach -t "$SESSION_NAME"
