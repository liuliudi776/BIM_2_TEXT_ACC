import sqlite3
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime
import os
import shutil

def evaluate_compliance_check(config):
    """
    评测合规性审查任务 - 三分类问题（合规、不合规、不适用）
    
    评测规则：
      - 将规则序号和IFC实体组合并作为样本标识
      - 比较预测的judgment_result与标注的judgment_result是否一致
      - 对于标注中存在但预测中缺失的样本：
        * 如果标注为"不适用"，视为预测正确
        * 否则视为预测错误
      - 对于预测中出现但标注中不存在的样本：
        * 预测为"不适用"视为正确
        * 预测为"合规"或"不合规"视为错误
    """
    # 构建数据库路径
    db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
    conn = sqlite3.connect(db_path)
  
    # 从原始预测表与标注表获取数据
    pred_query = "SELECT 规则序号, IFC实体组, judgment_result FROM 结果_9_合规性审查"
    true_query = "SELECT 规则序号, IFC实体组, judgment_result FROM 标注_合规性审查"
    
    try:
        pred_df = pd.read_sql_query(pred_query, conn)
        true_df = pd.read_sql_query(true_query, conn)
    except Exception as e:
        print(f"获取合规性审查数据时出错: {e}")
        # 如果表不存在，返回空结果
        return {
            'accuracy': 0,
            'macro_f1': 0, 
            'macro_recall': 0,
            'class_metrics': {},
            'total': 0,
            'details': [],
            'error_details': [],
            'confusion_matrix': {}
        }
  
    conn.close()
  
    # 填充空值，避免后续对比出错
    pred_df = pred_df.fillna('')
    true_df = true_df.fillna('')
    
    # 创建样本标识：规则序号+IFC实体组
    pred_df['sample_id'] = pred_df['规则序号'].astype(str) + "_" + pred_df['IFC实体组'].astype(str)
    true_df['sample_id'] = true_df['规则序号'].astype(str) + "_" + true_df['IFC实体组'].astype(str)
    
    # 将judgment_result规范化为三个类别（合规/不合规/不适用）
    def normalize_judgment(judgment):
        judgment = str(judgment).lower().strip()
        if judgment in ['合规', '符合', 'compliant', 'true', '是', 'yes', '1']:
            return '合规'
        elif judgment in ['不合规', '不符合', '违规', 'non-compliant', 'false', '否', 'no', '0']:
            return '不合规'
        else:
            return '不适用'
    
    pred_df['norm_judgment'] = pred_df['judgment_result'].apply(normalize_judgment)
    true_df['norm_judgment'] = true_df['judgment_result'].apply(normalize_judgment)
    
    # 创建预测结果和真实结果的字典，便于查询
    pred_dict = dict(zip(pred_df['sample_id'], pred_df['norm_judgment']))
    true_dict = dict(zip(true_df['sample_id'], true_df['norm_judgment']))
    
    # 合并所有样本ID
    all_samples = set(pred_dict.keys()).union(set(true_dict.keys()))
    
    # 初始化统计数据
    categories = ['合规', '不合规', '不适用']
    
    # 各类别的统计数据
    stats = {cat: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for cat in categories}
    
    # 详细结果记录
    details = []
    error_details = []  # 预测错误的样本
    
    total_samples = 0
    correct_samples = 0
    
    # 混淆矩阵
    confusion_matrix = {
        true_cat: {pred_cat: 0 for pred_cat in categories} 
        for true_cat in categories
    }
    
    # 分析每个样本
    for sample_id in all_samples:
        # 获取当前样本的预测和真实结果
        pred_judgment = pred_dict.get(sample_id, '未预测')  # 对于预测中不存在的样本，标记为"未预测"
        true_judgment = true_dict.get(sample_id, '未知')  # 对于标注中不存在的样本，标记为"未知"
        
        # 分解sample_id以获取规则序号和IFC实体组
        parts = sample_id.split('_', 1)
        rule_id = parts[0]
        ifc_group = parts[1] if len(parts) > 1 else ""
        
        # 处理缺失情况：
        # 1. 对于标注中不存在的样本（true_judgment为'未知'），将其视为"不适用"
        if true_judgment == '未知':
            true_judgment = '不适用'
            
        # 2. 对于预测中不存在的样本（pred_judgment为'未预测'）
        #    如果标注为"不适用"，视为正确预测"不适用"
        if pred_judgment == '未预测':
            if true_judgment == '不适用':
                pred_judgment = '不适用'  # 将未预测视为预测了"不适用"
            else:
                # 对于其他情况，保持为'未预测'
                pass
                
        # 计入总样本数（只计入有效样本）
        if pred_judgment != '未预测':
            total_samples += 1
            
        # 判断是否正确
        is_correct = pred_judgment == true_judgment
        if is_correct:
            correct_samples += 1
            
            # 更新各类别的TP
            for cat in categories:
                if true_judgment == cat and pred_judgment == cat:
                    stats[cat]['TP'] += 1
        else:
            # 更新FP和FN
            if pred_judgment in categories:
                stats[pred_judgment]['FP'] += 1
            if true_judgment in categories:
                stats[true_judgment]['FN'] += 1
        
        # 更新混淆矩阵（仅对于有效预测）
        if pred_judgment in categories:
            confusion_matrix[true_judgment][pred_judgment] += 1
        
        # 记录结果
        result = {
            '规则序号': rule_id,
            'IFC实体组': ifc_group,
            '预测结果': pred_judgment,
            '实际结果': true_judgment,
            '是否正确': is_correct
        }
        details.append(result)
        
        # 记录错误案例
        if not is_correct:
            error_details.append(result)
    
    # 计算每个类别的指标
    class_metrics = {}
    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    total_support = 0
    classes_with_samples = 0
  
    for category in categories:
        # 获取统计数据
        tp = stats[category]['TP']
        fp = stats[category]['FP']
        fn = stats[category]['FN']
        support = tp + fn  # 该类别的真实样本数
      
        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
      
        # 确保不会出现大于1的指标
        precision = min(1.0, precision)
        recall = min(1.0, recall)
        f1 = min(1.0, f1)
      
        class_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support  # 该类别的真实样本数
        }
      
        # 计算加权平均的分子部分
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        total_support += support
      
        # 只计入有样本的类别
        if support > 0:
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            classes_with_samples += 1
  
    # 计算宏平均指标
    if classes_with_samples > 0:
        macro_precision /= classes_with_samples
        macro_recall /= classes_with_samples
        macro_f1 /= classes_with_samples
  
    # 计算加权平均指标
    if total_support > 0:
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support
  
    # 计算准确率
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
  
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics,
        'total': total_samples,
        'correct': correct_samples,
        'details': details,
        'error_details': error_details,
        'confusion_matrix': confusion_matrix
    }
  
    return metrics

def create_evaluate_tables(config):
    """
    创建评测相关的视图
    """
    # 构建数据库路径，假设路径中包含 "data.db"，并根据当前模型类型替换相应部分
    db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
  
    # 连接数据库，并创建视图
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
  
        # 创建评测_规范类型评测视图
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS 评测_规范类型评测视图 AS
        SELECT 
            r.规则序号,
            r.规范内容,
            r.识别类型,
            l.识别类型 as 标注类型
        FROM 
            结果_1_规范类型识别 r
        LEFT JOIN 
            标注_规范类型 l
        ON 
            r.规则序号 = l.规则序号
        """)
  
        # 创建评测_规范实体识别评测视图（此视图供人工核查或其他用途；NER评测将使用原始表数据）
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS 评测_规范实体识别评测视图 AS
        SELECT 
            COALESCE(r.规则序号, l.规则序号) as 规则序号,
            r.实体文本,
            r.实体类型,
            r.开始位置,
            r.结束位置,
            l.实体文本 as 标注实体文本,
            l.实体类型 as 标注实体类型,
            l.开始位置 as 标注开始位置,
            l.结束位置 as 标注结束位置
        FROM 
            结果_2_规范实体识别 r
        FULL OUTER JOIN 
            标注_规范实体识别 l
        ON 
            r.规则序号 = l.规则序号
        """)
  
        # 创建评测_IFC实体对齐评测视图
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS 评测_IFC实体对齐评测视图 AS
        SELECT 
            COALESCE(r.规则序号, l.规则序号) as 规则序号,
            r.规范实体文本,
            r.规范实体类型,
            r.ifc_guid,
            r.ifc_entity_with_type,
            l.ifc_guid as 标注ifc_guid,
            l.ifc_entity_with_type as 标注ifc_entity_with_type
        FROM 
            结果_4_实体对齐 r
        FULL OUTER JOIN 
            标注_IFC实体对齐 l
        ON 
            r.规则序号 = l.规则序号
            AND r.ifc_guid = l.ifc_guid
        """)
  
        # 提交事务
        conn.commit()

def evaluate_ner_text_based(config):
    """
    使用基于文本的匹配策略评测 NER 任务（规范实体识别），不考虑位置信息。
  
    评测规则：
      - 对同一规则序号下的预测实体和标注实体进行匹配。
      - 匹配条件：
          * 实体类型必须一致；
          * 实体文本必须相同（忽略大小写和前后空格）。
      - 采用一对一匹配：一个标注实体匹配成功后，不再用于后续匹配。
  
    返回：
      - 总体指标: Precision, Recall 和 F1 Score
      - 详细的实体级别匹配情况，包括每个TP、FP和FN的具体信息
    """
    # 构建数据库路径
    db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
    conn = sqlite3.connect(db_path)
  
    # 从原始预测表与标注表获取数据
    pred_query = "SELECT 规则序号, 实体文本, 实体类型, 开始位置, 结束位置 FROM 结果_2_规范实体识别"
    true_query = "SELECT 规则序号, 实体文本, 实体类型, 开始位置, 结束位置 FROM 标注_规范实体识别"
    pred_df = pd.read_sql_query(pred_query, conn)
    true_df = pd.read_sql_query(true_query, conn)
  
    conn.close()
  
    # 填充空值，避免后续对比出错
    pred_df = pred_df.fillna('')
    true_df = true_df.fillna('')
  
    # 标准化实体文本（去除前后空格并转为小写）
    pred_df['norm_text'] = pred_df['实体文本'].str.strip().str.lower()
    true_df['norm_text'] = true_df['实体文本'].str.strip().str.lower()
  
    # 按规则序号分组
    grouped_pred = pred_df.groupby("规则序号")
    grouped_true = true_df.groupby("规则序号")
    rule_ids = set(pred_df["规则序号"]).union(set(true_df["规则序号"]))
  
    total_tp = 0   # 真正匹配成功数量
    total_pred = 0 # 预测实体数量
    total_true = 0 # 标注实体数量
  
    # 用于记录每个规则的详细匹配情况
    rule_details = []
  
    # 用于记录详细的实体匹配情况
    true_positives = []  # 成功匹配的实体对
    false_positives = []  # 多余预测的实体
    false_negatives = []  # 未被预测的实体
  
    # 遍历每个规则序号下的实体进行一对一匹配
    for rid in rule_ids:
        preds = grouped_pred.get_group(rid).to_dict('records') if rid in grouped_pred.groups else []
        trues = grouped_true.get_group(rid).to_dict('records') if rid in grouped_true.groups else []
      
        total_pred += len(preds)
        total_true += len(trues)
      
        # 创建标记数组来跟踪哪些实体已被匹配
        matched_trues = [False] * len(trues)
        matched_preds = [False] * len(preds)
        tp_in_rule = 0
        
        # 第一次遍历：尝试完全匹配（实体类型和文本都一致）
        for p_idx, p in enumerate(preds):
            if matched_preds[p_idx]:
                continue
                
            for t_idx, t in enumerate(trues):
                if matched_trues[t_idx]:
                    continue
                    
                # 检查类型和规范化后的文本是否完全匹配
                if p['实体类型'] == t['实体类型'] and p['norm_text'] == t['norm_text']:
                    matched_preds[p_idx] = True
                    matched_trues[t_idx] = True
                    tp_in_rule += 1
                    
                    # 记录匹配成功的实体对
                    true_positives.append({
                        '规则序号': rid,
                        '预测实体': p["实体文本"],
                        '预测类型': p["实体类型"],
                        '预测开始位置': p["开始位置"],
                        '预测结束位置': p["结束位置"],
                        '标注实体': t["实体文本"],
                        '标注类型': t["实体类型"],
                        '标注开始位置': t["开始位置"],
                        '标注结束位置': t["结束位置"],
                        '匹配方式': '完全匹配'
                    })
                    break
        
        # 第二次遍历：尝试部分文本匹配（实体类型相同，一个文本包含另一个）
        for p_idx, p in enumerate(preds):
            if matched_preds[p_idx]:
                continue
                
            for t_idx, t in enumerate(trues):
                if matched_trues[t_idx]:
                    continue
                    
                # 检查类型是否匹配，且文本有部分重叠
                if p['实体类型'] == t['实体类型'] and (
                   p['norm_text'] in t['norm_text'] or t['norm_text'] in p['norm_text']):
                    matched_preds[p_idx] = True
                    matched_trues[t_idx] = True
                    tp_in_rule += 1
                    
                    # 记录匹配成功的实体对
                    true_positives.append({
                        '规则序号': rid,
                        '预测实体': p["实体文本"],
                        '预测类型': p["实体类型"],
                        '预测开始位置': p["开始位置"],
                        '预测结束位置': p["结束位置"],
                        '标注实体': t["实体文本"],
                        '标注类型': t["实体类型"],
                        '标注开始位置': t["开始位置"],
                        '标注结束位置': t["结束位置"],
                        '匹配方式': '部分匹配'
                    })
                    break
      
        total_tp += tp_in_rule
      
        # 记录未匹配的预测实体（假阳性/FP）
        for p_idx, p in enumerate(preds):
            if not matched_preds[p_idx]:
                false_positives.append({
                    '规则序号': rid,
                    '预测实体': p["实体文本"],
                    '预测类型': p["实体类型"],
                    '预测开始位置': p["开始位置"],
                    '预测结束位置': p["结束位置"]
                })
      
        # 记录未匹配的标注实体（假阴性/FN）
        for t_idx, t in enumerate(trues):
            if not matched_trues[t_idx]:
                false_negatives.append({
                    '规则序号': rid,
                    '标注实体': t["实体文本"],
                    '标注类型': t["实体类型"],
                    '标注开始位置': t["开始位置"],
                    '标注结束位置': t["结束位置"]
                })
      
        # 记录当前规则的详细情况
        rule_details.append({
            '规则序号': rid,
            '预测实体数': len(preds),
            '标注实体数': len(trues),
            '匹配成功数': tp_in_rule,
            '假阳性数': len(preds) - tp_in_rule,
            '假阴性数': len(trues) - tp_in_rule
        })
  
    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
  
    metrics = {
        'tp': total_tp,
        'fp': len(false_positives),
        'fn': len(false_negatives),
        'pred': total_pred,
        'true': total_true,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rule_details': rule_details,  # 每个规则的汇总情况
        'tp_details': true_positives,  # 详细的成功匹配实体对
        'fp_details': false_positives,  # 详细的假阳性实体
        'fn_details': false_negatives   # 详细的假阴性实体
    }
  
    return metrics

def evaluate_ifc_alignment(config):
    """
    评测IFC实体对齐任务。
    
    评测规则：
      - 对同一规则序号下的预测IFC对齐和标注IFC对齐进行匹配
      - 匹配条件是两者的ifc_guid完全一致
      - 关注点是规范条目下的IFC实体GUID集合是否正确识别
      
    返回：
      - 总体指标：Precision, Recall和F1 Score
      - 详细的匹配情况
    """
    # 构建数据库路径
    db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
    conn = sqlite3.connect(db_path)
  
    # 从原始预测表与标注表获取数据
    pred_query = "SELECT 规则序号, 规范实体文本, 规范实体类型, ifc_guid, ifc_entity_with_type FROM 结果_4_实体对齐"
    true_query = "SELECT 规则序号, 规范实体文本, 规范实体类型, ifc_guid, ifc_entity_with_type FROM 标注_IFC实体对齐"
    
    pred_df = pd.read_sql_query(pred_query, conn)
    true_df = pd.read_sql_query(true_query, conn)
  
    conn.close()
  
    # 填充空值，避免后续对比出错
    pred_df = pred_df.fillna('')
    true_df = true_df.fillna('')
    
    # 过滤掉空的ifc_guid
    pred_df = pred_df[pred_df['ifc_guid'].str.strip() != '']
    true_df = true_df[true_df['ifc_guid'].str.strip() != '']
    
    # 标准化ifc_guid (去除空格并转为小写)
    pred_df['norm_guid'] = pred_df['ifc_guid'].str.strip().str.lower()
    true_df['norm_guid'] = true_df['ifc_guid'].str.strip().str.lower()
  
    # 按规则序号分组
    grouped_pred = pred_df.groupby("规则序号")
    grouped_true = true_df.groupby("规则序号")
    rule_ids = set(pred_df["规则序号"]).union(set(true_df["规则序号"]))
  
    total_tp = 0   # 真正匹配成功数量
    total_pred = 0 # 预测对齐数量
    total_true = 0 # 标注对齐数量
  
    # 用于记录每个规则的详细匹配情况
    rule_details = []
  
    # 用于记录详细的IFC对齐匹配情况
    true_positives = []  # 成功匹配的对齐对
    false_positives = []  # 多余预测的对齐
    false_negatives = []  # 未被预测的对齐
  
    # 遍历每个规则序号下的IFC对齐进行匹配
    for rid in rule_ids:
        # 获取当前规则的预测和标注数据
        preds = grouped_pred.get_group(rid).to_dict('records') if rid in grouped_pred.groups else []
        trues = grouped_true.get_group(rid).to_dict('records') if rid in grouped_true.groups else []
        
        # 获取预测和标注的GUID集合
        pred_guids = set([p['norm_guid'] for p in preds])
        true_guids = set([t['norm_guid'] for t in trues])
        
        # 计算匹配数量
        tp_guids = pred_guids.intersection(true_guids)
        fp_guids = pred_guids - true_guids
        fn_guids = true_guids - pred_guids
        
        total_pred += len(pred_guids)
        total_true += len(true_guids)
        total_tp += len(tp_guids)
        
        # 记录当前规则的详细情况
        rule_details.append({
            '规则序号': rid,
            '预测对齐数': len(pred_guids),
            '标注对齐数': len(true_guids),
            '匹配成功数': len(tp_guids),
            '假阳性数': len(fp_guids),
            '假阴性数': len(fn_guids)
        })
        
        # 记录真阳性详情
        for guid in tp_guids:
            # 找到对应的预测和标注记录
            pred_item = next((p for p in preds if p['norm_guid'] == guid), None)
            true_item = next((t for t in trues if t['norm_guid'] == guid), None)
            
            if pred_item and true_item:
                true_positives.append({
                    '规则序号': rid,
                    '预测规范实体': pred_item["规范实体文本"],
                    '预测规范实体类型': pred_item["规范实体类型"],
                    '预测IFC_GUID': pred_item["ifc_guid"],
                    '预测IFC实体': pred_item["ifc_entity_with_type"],
                    '标注规范实体': true_item["规范实体文本"],
                    '标注规范实体类型': true_item["规范实体类型"],
                    '标注IFC_GUID': true_item["ifc_guid"],
                    '标注IFC实体': true_item["ifc_entity_with_type"]
                })
        
        # 记录假阳性详情
        for guid in fp_guids:
            pred_item = next((p for p in preds if p['norm_guid'] == guid), None)
            if pred_item:
                false_positives.append({
                    '规则序号': rid,
                    '预测规范实体': pred_item["规范实体文本"],
                    '预测规范实体类型': pred_item["规范实体类型"],
                    '预测IFC_GUID': pred_item["ifc_guid"],
                    '预测IFC实体': pred_item["ifc_entity_with_type"]
                })
        
        # 记录假阴性详情
        for guid in fn_guids:
            true_item = next((t for t in trues if t['norm_guid'] == guid), None)
            if true_item:
                false_negatives.append({
                    '规则序号': rid,
                    '标注规范实体': true_item["规范实体文本"],
                    '标注规范实体类型': true_item["规范实体类型"],
                    '标注IFC_GUID': true_item["ifc_guid"],
                    '标注IFC实体': true_item["ifc_entity_with_type"]
                })
  
    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
  
    metrics = {
        'tp': total_tp,
        'fp': len(false_positives),
        'fn': len(false_negatives),
        'pred': total_pred,
        'true': total_true,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rule_details': rule_details,  # 每个规则的汇总情况
        'tp_details': true_positives,  # 详细的成功匹配对齐对
        'fp_details': false_positives,  # 详细的假阳性对齐
        'fn_details': false_negatives   # 详细的假阴性对齐
    }
  
    return metrics

def generate_report(config):
    """
    生成评测报告，包括：
    1. Accuracy、F1 Score 和 Recall 等常规指标
    2. NER 任务的详细错误分析（包含假阳性、假阴性、真阳性具体实例）
    3. IFC实体对齐的详细错误分析
    """
    db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
    conn = sqlite3.connect(db_path)
  
    evaluation_views = {
        "评测_规范类型评测视图": {"pred": "识别类型", "true": "标注类型"},
    }
  
    report = {}
  
    for view_key, columns in evaluation_views.items():
        query = f"SELECT {columns['pred']} as pred, {columns['true']} as true FROM {view_key}"
        df = pd.read_sql_query(query, conn)
  
        if df.empty:
            print(f"视图 {view_key} 中没有数据，跳过评测!")
            continue
  
        df['pred'] = df['pred'].fillna("NaN")
        df['true'] = df['true'].fillna("NaN")
  
        acc = accuracy_score(df['true'], df['pred'])
        f1 = f1_score(df['true'], df['pred'], average='macro')
        recall_metric = recall_score(df['true'], df['pred'], average='macro', zero_division=0)
  
        report[view_key] = {"accuracy": acc, "f1": f1, "recall": recall_metric}
  
    conn.close()
  
    console = Console()
  
    main_table = Table(title="评测结果汇总", box=box.ROUNDED)
    main_table.add_column("评测项", style="cyan")
    main_table.add_column("Accuracy", justify="right", style="green")
    main_table.add_column("F1 Score", justify="right", style="green")
    main_table.add_column("Recall", justify="right", style="green")
  
    for key, metrics in report.items():
        acc = f"{metrics['accuracy']:.4f}"
        f1 = metrics['f1'] if isinstance(metrics['f1'], str) else f"{metrics['f1']:.4f}"
        recall = metrics['recall'] if isinstance(metrics['recall'], str) else f"{metrics['recall']:.4f}"
        main_table.add_row(key, acc, f1, recall)
  
    # 使用基于文本的NER评测
    ner_metrics = evaluate_ner_text_based(config)
    main_table.add_row("NER (文本匹配)", f"{ner_metrics['precision']:.4f}", f"{ner_metrics['f1']:.4f}", f"{ner_metrics['recall']:.4f}")
    
    # 使用IFC实体对齐评测
    ifc_metrics = evaluate_ifc_alignment(config)
    main_table.add_row("IFC实体对齐", f"{ifc_metrics['precision']:.4f}", f"{ifc_metrics['f1']:.4f}", f"{ifc_metrics['recall']:.4f}")
  
    # 使用合规性审查评测
    compliance_metrics = evaluate_compliance_check(config)
    if compliance_metrics['total'] > 0:
        main_table.add_row("合规性审查", f"{ifc_metrics['precision']:.4f}", f"{ifc_metrics['f1']:.4f}", f"{ifc_metrics['recall']:.4f}")

  
    console.print(main_table)
    console.print()

    filename = f"{config['current_model']['type'].replace(' ', '_')}_evaluate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    if config["building_type"]["type"] == "桥梁":
        if os.path.exists(f"evaluate/evaluate_bridge_results_dir/{config['current_model']['type']}"):
            shutil.rmtree(f"evaluate/evaluate_bridge_results_dir/{config['current_model']['type']}")
        os.makedirs(f"evaluate/evaluate_bridge_results_dir/{config['current_model']['type']}", exist_ok=True)
        filename = os.path.join(f"evaluate/evaluate_bridge_results_dir/{config['current_model']['type']}", filename)
    else:
        if os.path.exists(f"evaluate/evaluate_building_results_dir/{config['current_model']['type']}"):
            shutil.rmtree(f"evaluate/evaluate_building_results_dir/{config['current_model']['type']}")
        os.makedirs(f"evaluate/evaluate_building_results_dir/{config['current_model']['type']}", exist_ok=True)
        filename = os.path.join(f"evaluate/evaluate_building_results_dir/{config['current_model']['type']}", filename)
  
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# 评测结果汇总\n")
      
        f.write("## 主要评测指标\n")
        f.write("| 评测项 | Accuracy | F1 Score | Recall |\n")
        f.write("|--------|----------|-----------|--------|\n")
      
        for key, metrics in report.items():
            acc = f"{metrics['accuracy']:.4f}"
            f1 = metrics['f1'] if isinstance(metrics['f1'], str) else f"{metrics['f1']:.4f}"
            recall = metrics['recall'] if isinstance(metrics['recall'], str) else f"{metrics['recall']:.4f}"
            f.write(f"| {key} | {acc} | {f1} | {recall} |\n")
      
        f.write(f"| NER (文本匹配) | {ner_metrics['precision']:.4f} | {ner_metrics['f1']:.4f} | {ner_metrics['recall']:.4f} |\n")
        f.write(f"| IFC实体对齐 | {ifc_metrics['precision']:.4f} | {ifc_metrics['f1']:.4f} | {ifc_metrics['recall']:.4f} |\n\n")

        # 写入NER详细评测结果
        f.write("## NER 详细指标\n")
        f.write("| 规则序号 | 预测实体数 | 标注实体数 | 匹配成功数 | 假阳性数 | 假阴性数 |\n")
        f.write("|---------|------------|------------|------------|----------|----------|\n")

        for rule_detail in ner_metrics['rule_details']:
            f.write(f"| {rule_detail['规则序号']} | {rule_detail['预测实体数']} | " +
                    f"{rule_detail['标注实体数']} | {rule_detail['匹配成功数']} | " +
                    f"{rule_detail['假阳性数']} | {rule_detail['假阴性数']} |\n")

        f.write("\n## NER 详细错误分析\n")

        # 写入真阳性详情
        f.write("### 成功匹配的实体 (True Positives)\n\n")
        if ner_metrics['tp_details']:
            f.write("| 规则序号 | 预测实体 | 预测类型 | 预测位置 | 标注实体 | 标注类型 | 标注位置 | 匹配方式 |\n")
            f.write("|---------|----------|----------|----------|----------|----------|----------|----------|\n")
            for tp in ner_metrics['tp_details']:
                p_pos = f"{tp['预测开始位置']}-{tp['预测结束位置']}"
                t_pos = f"{tp['标注开始位置']}-{tp['标注结束位置']}"
                match_type = tp.get('匹配方式', '普通匹配')
                f.write(f"| {tp['规则序号']} | {tp['预测实体']} | {tp['预测类型']} | {p_pos} | {tp['标注实体']} | {tp['标注类型']} | {t_pos} | {match_type} |\n")
        else:
            f.write("无成功匹配的实体。\n")
        f.write("\n")

        # 写入假阳性详情
        f.write("### 多余预测的实体 (False Positives)\n\n")
        if ner_metrics['fp_details']:
            f.write("| 规则序号 | 预测实体 | 预测类型 | 预测位置 |\n")
            f.write("|---------|----------|----------|----------|\n")
            for fp in ner_metrics['fp_details']:
                p_pos = f"{fp['预测开始位置']}-{fp['预测结束位置']}"
                f.write(f"| {fp['规则序号']} | {fp['预测实体']} | {fp['预测类型']} | {p_pos} |\n")
        else:
            f.write("无多余预测的实体。\n")
        f.write("\n")

        # 写入假阴性详情
        f.write("### 未被预测的实体 (False Negatives)\n\n")
        if ner_metrics['fn_details']:
            f.write("| 规则序号 | 标注实体 | 标注类型 | 标注位置 |\n")
            f.write("|---------|----------|----------|----------|\n")
            for fn in ner_metrics['fn_details']:
                t_pos = f"{fn['标注开始位置']}-{fn['标注结束位置']}"
                f.write(f"| {fn['规则序号']} | {fn['标注实体']} | {fn['标注类型']} | {t_pos} |\n")
        else:
            f.write("无未被预测的实体。\n")
        f.write("\n")

        # 写入IFC实体对齐详细评测结果
        f.write("## IFC实体对齐详细指标\n")
        f.write("| 规则序号 | 预测对齐数 | 标注对齐数 | 匹配成功数 | 假阳性数 | 假阴性数 |\n")
        f.write("|---------|------------|------------|------------|----------|----------|\n")

        for rule_detail in ifc_metrics['rule_details']:
            f.write(f"| {rule_detail['规则序号']} | {rule_detail['预测对齐数']} | " +
                    f"{rule_detail['标注对齐数']} | {rule_detail['匹配成功数']} | " +
                    f"{rule_detail['假阳性数']} | {rule_detail['假阴性数']} |\n")

        f.write("\n## IFC实体对齐详细错误分析\n")

        # 写入IFC对齐真阳性详情
        f.write("### 成功匹配的IFC对齐 (True Positives)\n\n")
        if ifc_metrics['tp_details']:
            f.write("| 规则序号 | 预测规范实体 | 预测实体类型 | 预测IFC_GUID | 预测IFC实体 | 标注规范实体 | 标注实体类型 | 标注IFC_GUID | 标注IFC实体 |\n")
            f.write("|---------|------------|------------|-------------|-----------|------------|------------|-------------|-------------|\n")
            for tp in ifc_metrics['tp_details']:
                f.write(f"| {tp['规则序号']} | {tp['预测规范实体']} | {tp['预测规范实体类型']} | {tp['预测IFC_GUID']} | {tp['预测IFC实体']} | " +
                        f"{tp['标注规范实体']} | {tp['标注规范实体类型']} | {tp['标注IFC_GUID']} | {tp['标注IFC实体']} |\n")
        else:
            f.write("无成功匹配的IFC对齐。\n")
        f.write("\n")

        # 写入IFC对齐假阳性详情
        f.write("### 多余预测的IFC对齐 (False Positives)\n\n")
        if ifc_metrics['fp_details']:
            f.write("| 规则序号 | 预测规范实体 | 预测实体类型 | 预测IFC_GUID | 预测IFC实体 |\n")
            f.write("|---------|------------|------------|-------------|-------------|\n")
            for fp in ifc_metrics['fp_details']:
                f.write(f"| {fp['规则序号']} | {fp['预测规范实体']} | {fp['预测规范实体类型']} | {fp['预测IFC_GUID']} | {fp['预测IFC实体']} |\n")
        else:
            f.write("无多余预测的IFC对齐。\n")
        f.write("\n")

        # 写入IFC对齐假阴性详情
        f.write("### 未被预测的IFC对齐 (False Negatives)\n\n")
        if ifc_metrics['fn_details']:
            f.write("| 规则序号 | 标注规范实体 | 标注实体类型 | 标注IFC_GUID | 标注IFC实体 |\n")
            f.write("|---------|------------|------------|-------------|-------------|\n")
            for fn in ifc_metrics['fn_details']:
                f.write(f"| {fn['规则序号']} | {fn['标注规范实体']} | {fn['标注规范实体类型']} | {fn['标注IFC_GUID']} | {fn['标注IFC实体']} |\n")
        else:
            f.write("无未被预测的IFC对齐。\n")
        f.write("\n")

        if compliance_metrics['total'] > 0:
            f.write(f"| 合规性审查 | {compliance_metrics['accuracy']:.4f} | {compliance_metrics['macro_f1']:.4f} | {compliance_metrics['macro_recall']:.4f} |\n\n")
            # 写入合规性审查详细评测结果
            f.write("## 合规性审查详细指标\n")
            f.write(f"- 总样本数: {compliance_metrics['total']}\n")
            f.write(f"- 正确预测数: {compliance_metrics['correct']}\n")
            f.write(f"- 准确率: {compliance_metrics['accuracy']:.4f}\n")
            f.write(f"- 宏平均精确率: {compliance_metrics['macro_precision']:.4f}\n")
            f.write(f"- 宏平均召回率: {compliance_metrics['macro_recall']:.4f}\n")
            f.write(f"- 宏平均F1分数: {compliance_metrics['macro_f1']:.4f}\n")
            f.write(f"- 加权平均精确率: {compliance_metrics['weighted_precision']:.4f}\n")
            f.write(f"- 加权平均召回率: {compliance_metrics['weighted_recall']:.4f}\n")
            f.write(f"- 加权平均F1分数: {compliance_metrics['weighted_f1']:.4f}\n\n")
            
            # 写入各类别的详细指标
            f.write("### 各类别详细指标\n\n")
            f.write("| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
          
            for category, metrics in compliance_metrics['class_metrics'].items():
                f.write(f"| {category} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |\n")
            f.write("\n")
          
            # 写入混淆矩阵
            f.write("### 混淆矩阵\n\n")
            f.write("| 实际\\预测 | 合规 | 不合规 | 不适用 |\n")
            f.write("|-----------|------|--------|--------|\n")
          
            for true_cat in ['合规', '不合规', '不适用']:
                row = f"| {true_cat} | "
                for pred_cat in ['合规', '不合规', '不适用']:
                    row += f"{compliance_metrics['confusion_matrix'][true_cat][pred_cat]} | "
                f.write(row + "\n")
            f.write("\n")
            
            # 写入错误预测详情
            f.write("### 合规性审查预测错误案例\n\n")
            if compliance_metrics['error_details']:
                f.write("| 规则序号 | IFC实体组 | 预测结果 | 实际结果 |\n")
                f.write("|---------|-----------|----------|----------|\n")
                for error in compliance_metrics['error_details']:
                    f.write(f"| {error['规则序号']} | {error['IFC实体组']} | {error['预测结果']} | {error['实际结果']} |\n")
            else:
                f.write("无预测错误案例。\n")
            f.write("\n")

    console.print(f"评测报告已保存至: {filename}")

def main(config):
    # 先创建所有评测视图
    create_evaluate_tables(config)
    # 再生成评测报告（包含常规指标与文本匹配的NER评测和IFC实体对齐评测）
    generate_report(config)
