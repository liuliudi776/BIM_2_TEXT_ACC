"""
规范合规性审查系统主程序
"""

import datetime
from pathlib import Path
import tomli
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import sys
import subprocess
import os

from utils.utils_control_logger import control_logger as logger
import utils.utils_completion_notion as utils_completion_notion
from modules.database import init_database
from modules.rule_processing import recognize_rule_types, recognize_ner_candidates, align_entities, recognize_inclusion_relationships
from modules.ifc_processing import extract_ifc_entities, extract_ifc_properties
from modules.building_ifc_element_path import main as extract_building_ifc_element_path, calculate_rule_element_groups
from modules.bridge_ifc_element_path import main as extract_bridge_ifc_element_path
from modules.property_strategy import save_property_selection_strategy, save_relevant_properties_with_strategy
from modules.nlp_generation import generate_natural_language_descriptions, check_compliance
from modules.report_table import main as create_report_views
from modules.evaluate import main as evaluate

def main(config):
    
    console = Console()
    total_start_time = datetime.datetime.now()
    
    logger.info(f"开始执行工作流程: {total_start_time}")
    step_times = {}

    # 获取步骤控制标志
    step_controls = config.get("step_controls", {
        "init_database": 0, #1
        "recognize_rule_types": 0, #2
        "recognize_ner": 0, #3
        "recognize_entity_relations": 0, #4
        "extract_ifc_entities": 0, #5
        "extract_ifc_element_path": 0, #6
        "align_entities": 0, #7
        "calculate_rule_element_groups": 0, #8
        "extract_ifc_properties": 0, #9
        "save_property_strategy": 0, #10
        "save_relevant_properties": 0, #11
        "generate_descriptions": 0, #12
        "check_compliance": 0, #13
        "create_report_views": 0, #14
        "evaluate": 0 #15
    })

    try:
        # 1. 初始化数据库
        if step_controls.get("init_database", 1):
            start_time = datetime.datetime.now()
            init_database(config)
            step_times['初始化数据库'] = datetime.datetime.now() - start_time
        
        # 2. 识别规范类型
        if step_controls.get("recognize_rule_types", 1):
            start_time = datetime.datetime.now()
            recognize_rule_types(config)
            step_times['识别规范类型'] = datetime.datetime.now() - start_time

        # 3. 识别命名实体
        if step_controls.get("recognize_ner", 1):
            start_time = datetime.datetime.now()
            recognize_ner_candidates(config)
            step_times['识别命名实体'] = datetime.datetime.now() - start_time

        # 4. 识别实体关系
        if step_controls.get("recognize_entity_relations", 1):
            start_time = datetime.datetime.now()
            recognize_inclusion_relationships(config)
            step_times['识别实体关系'] = datetime.datetime.now() - start_time

        # 5. 提取IFC实体信息
        if step_controls.get("extract_ifc_entities", 1):
            start_time = datetime.datetime.now()
            extract_ifc_entities(config)
            step_times['提取IFC实体'] = datetime.datetime.now() - start_time

        # 6. 提取IFC元素路径
        if step_controls.get("extract_ifc_element_path", 1):
            if config['building_type']['type'] == '桥梁':
                start_time = datetime.datetime.now()
                extract_bridge_ifc_element_path(config)
                step_times['提取IFC元素路径'] = datetime.datetime.now() - start_time
            else:
                start_time = datetime.datetime.now()
                extract_building_ifc_element_path(config)
                step_times['提取IFC元素路径'] = datetime.datetime.now() - start_time

        # 7. 实体对齐
        if step_controls.get("align_entities", 1):
            start_time = datetime.datetime.now()
            align_entities(config)
            step_times['实体对齐'] = datetime.datetime.now() - start_time

        # 8. 计算规范元素组
        if step_controls.get("calculate_rule_element_groups", 1):
            start_time = datetime.datetime.now()
            calculate_rule_element_groups(config)
            step_times['计算规范元素组'] = datetime.datetime.now() - start_time

        # 9. 提取IFC属性集
        if step_controls.get("extract_ifc_properties", 1):
            start_time = datetime.datetime.now()
            extract_ifc_properties(config)
            step_times['提取IFC属性集'] = datetime.datetime.now() - start_time

        # 10. 保存属性选择策略
        if step_controls.get("save_property_strategy", 1):
            start_time = datetime.datetime.now()
            save_property_selection_strategy(config)
            step_times['保存属性选择策略'] = datetime.datetime.now() - start_time

        # 11. 选择相关属性
        if step_controls.get("save_relevant_properties", 1):
            start_time = datetime.datetime.now()
            save_relevant_properties_with_strategy(config)
            step_times['选择相关属性'] = datetime.datetime.now() - start_time

        # 12. 生成自然语言描述
        if step_controls.get("generate_descriptions", 1):
            start_time = datetime.datetime.now()
            generate_natural_language_descriptions(config)
            step_times['生成自然语言描述'] = datetime.datetime.now() - start_time

        # 13. 合规性审查
        if step_controls.get("check_compliance", 1):
            start_time = datetime.datetime.now()
            check_compliance(config)
            step_times['合规性审查'] = datetime.datetime.now() - start_time

        # 14. 创建报告相关的视图
        if step_controls.get("create_report_views", 1):
            start_time = datetime.datetime.now()
            create_report_views(config)
            step_times['创建报告视图'] = datetime.datetime.now() - start_time

        # 15. 评测
        if step_controls.get("evaluate", 1):
            start_time = datetime.datetime.now()
            evaluate(config)
            step_times['评测'] = datetime.datetime.now() - start_time

        # 计算总执行时间
        total_time = datetime.datetime.now() - total_start_time

        # 创建统计表格
        table = Table(title="执行时间统计", show_header=True, header_style="bold magenta")
        table.add_column("步骤", style="cyan", no_wrap=True)
        table.add_column("执行时间", justify="right", style="green")
        table.add_column("占比", justify="right", style="yellow")

        # 添加每个步骤的统计数据
        for step, duration in step_times.items():
            percentage = (duration.total_seconds() / total_time.total_seconds()) * 100
            table.add_row(
                step,
                str(duration),
                f"{percentage:.1f}%"
            )

        # 添加总计行
        table.add_row(
            "总计",
            str(total_time),
            "100%",
            style="bold"
        )

        # 输出表格
        console.print("\n")
        console.print(Panel(
            table,
            title="[bold blue]工作流程执行完成",
            subtitle=f"[bold green]开始时间: {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ))
        
        logger.success("工作流程执行完成")
        
        # 调用模块中的方法发送通知
        # utils_completion_notion.send_completion_notification(total_time)

        # 执行完成后发送ctrl+d关闭tmux
        try:
            # 检查是否在tmux会话中
            if os.environ.get('TMUX'):
                # 发送ctrl+d组合键到tmux
                subprocess.run(['tmux', 'send-keys', 'C-d'])
        except Exception as e:
            print(f"关闭tmux时发生错误: {e}")

    except Exception as e:
        error_text = Text(f"工作流程执行失败: {str(e)}", style="bold red")
        console.print(Panel(error_text, title="错误"))
        logger.error(f"工作流程执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='规范合规性审查系统')
    parser.add_argument('--config', type=str, help='单个配置文件路径')
    args = parser.parse_args()
    
    if args.config:
        # 如果指定了配置文件,则只处理该文件
        with open(args.config, "rb") as f:
            config = tomli.load(f)
        # 加载gpt配置
        with open("config/openai_api.toml", "rb") as f:
            config["gpt_config"] = tomli.load(f)
        main(config)
    else:
        # 否则处理默认配置文件列表
        config_files = [
            # "config/bridge_config-Qwen2.5-72B-Instruct.toml",
            "config/bridge_config-Qwen2.5-32B-Instruct.toml",
            # "config/bridge_config-Qwen2.5-14B-Instruct.toml",
            # "config/bridge_config-Qwen2.5-7B-Instruct.toml",
            # "config/bridge_config-4o-mini.toml",
            # "config/bridge_config-deepseek-v3.toml",
            # # 建筑
            # "config/building_config-Qwen2.5-72B-Instruct.toml",
            "config/building_config-Qwen2.5-32B-Instruct.toml",
            # "config/building_config-Qwen2.5-14B-Instruct.toml",
            # "config/building_config-Qwen2.5-7B-Instruct.toml",
            # "config/building_config-4o-mini.toml",
            # "config/building_config-deepseek-v3.toml",
        ]

        # 加载所有配置文件
        CONFIG_LIST = []
        for config_file in config_files:
            with open(config_file, "rb") as f:
                CONFIG_LIST.append(tomli.load(f))

        # 加载gpt配置
        with open("config/openai_api.toml", "rb") as f:
            GPT_CONFIG = tomli.load(f)

        for config in CONFIG_LIST:
            config["gpt_config"] = GPT_CONFIG
            main(config)