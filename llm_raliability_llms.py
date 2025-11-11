# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:14:05 2025

@author: Fei Wang
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
import datetime
import time
import gc
from sentence_transformers import SentenceTransformer

# 配置路径
outpath = 'E:/Rdaima/RIS_theory/llm_raliability'
llm_models_dir = 'C:/workship/llm_models'
open_input_file = 'E:/Rdaima/RIS_theory/embedding_second/第二次合并后的编码结果.csv'
axial_input_file = 'E:/Rdaima/RIS_theory/Axial_output2/轴心编码结果.csv'

def load_embedding_model():
    """加载embedding模型"""
    model_file = os.path.join(llm_models_dir, "qwen3-embedding-0.6B")
    
    print(f"开始载入大语言模型文件 {model_file} ...")
    print("当前时间:", datetime.datetime.now())
    tbgn = time.time()
    model = SentenceTransformer(model_file)
    print(f"载入完毕，耗费时间：{(time.time() - tbgn) / 60:.2f} minutes")
    return model

def clean_text(text):
    """清理文本，移除文件标记等无关内容"""
    if pd.isna(text):
        return ""
    
    # 移除文件标记，如 [文件: sub008.txt]
    text = re.sub(r'\[文件:\s*[^]]+\]', '', text)
    # 移除其他可能的标记
    text = re.sub(r'<br>', ' ', text)
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_statements(text):
    """从具体语句汇总中提取独立的语句"""
    if pd.isna(text):
        return []
    
    # 清理文本
    cleaned_text = clean_text(text)
    
    # 按分隔符分割语句
    statements = re.split(r'\|', cleaned_text)
    
    # 进一步清理每个语句
    cleaned_statements = []
    for stmt in statements:
        stmt = stmt.strip()
        # 移除引号和其他可能的分隔符
        stmt = re.sub(r'^["\']|["\']$', '', stmt)
        if stmt and len(stmt) > 5:  # 过滤掉太短的语句
            cleaned_statements.append(stmt)
    
    return cleaned_statements

def calculate_reliability_metrics(embeddings):
    """计算信度指标 - 正确版本"""
    n = len(embeddings)
    
    if n < 2:
        return {
            'cronbach_alpha': 0.0,
            'average_interitem_correlation': 0.0,
            'item_count': n
        }
    
    try:
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 计算平均项目间相关（排除对角线）
        mask = ~np.eye(n, dtype=bool)
        avg_correlation = np.mean(similarity_matrix[mask])
        
        # 使用标准化的克隆巴赫Alpha公式
        # α = (n * r̄) / (1 + (n - 1) * r̄)
        # 其中r̄是平均项目间相关
        if avg_correlation > 0:
            alpha = (n * avg_correlation) / (1 + (n - 1) * avg_correlation)
        else:
            alpha = 0.0
        
        return {
            'cronbach_alpha': max(0, min(1, alpha)),
            'average_interitem_correlation': max(0, min(1, avg_correlation)),
            'item_count': n
        }
        
    except Exception as e:
        print(f"计算信度指标时出错: {e}")
        return {
            'cronbach_alpha': 0.0,
            'average_interitem_correlation': 0.0,
            'item_count': n
        }

def analyze_open_coding_reliability_statements(model, open_coding_df):
    """分析开放式编码的信度 - 基于具体语句汇总"""
    print("开始分析开放式编码信度（基于具体语句）...")
    
    results = []
    
    for idx, row in open_coding_df.iterrows():
        coding_name = row['编码名称']
        statements_text = row.get('具体语句汇总', '')
        
        if pd.isna(statements_text) or not statements_text:
            print(f"编码 '{coding_name}': 没有具体语句汇总，跳过")
            continue
        
        # 提取具体语句
        statements = extract_statements(statements_text)
        
        if len(statements) < 2:
            print(f"编码 '{coding_name}': 语句数量不足 ({len(statements)})，跳过计算")
            continue
        
        print(f"处理编码 '{coding_name}': {len(statements)} 条具体语句")
        
        try:
            # 使用具体语句生成embedding
            embeddings = model.encode(statements, show_progress_bar=False, normalize_embeddings=True)
            
            # 计算信度指标
            reliability_metrics = calculate_reliability_metrics(embeddings)
            
            results.append({
                '编码类型': '开放式编码_具体语句',
                '编码名称': coding_name,
                '语句数量': reliability_metrics['item_count'],
                '克隆巴赫Alpha': reliability_metrics['cronbach_alpha'],
                '平均项目间相关': reliability_metrics['average_interitem_correlation'],
                '来源文件数量': row.get('来源文件数量', 'N/A'),
                '总出现次数': row.get('总出现次数', 'N/A'),
                '语句示例': '; '.join(statements[:2]) + ('...' if len(statements) > 2 else '')
            })
            
            print(f"  → Alpha = {reliability_metrics['cronbach_alpha']:.4f}, 平均相关 = {reliability_metrics['average_interitem_correlation']:.4f}")
            
        except Exception as e:
            print(f"  → 处理编码 '{coding_name}' 时出错: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_open_coding_reliability_merged(model, open_coding_df):
    """分析开放式编码的信度 - 基于被合并的原始编码"""
    print("开始分析开放式编码信度（基于被合并编码）...")
    
    results = []
    
    for idx, row in open_coding_df.iterrows():
        coding_name = row['编码名称']
        merged_codings = row.get('被合并的原始编码', '')
        
        if pd.isna(merged_codings) or not merged_codings:
            print(f"编码 '{coding_name}': 没有被合并的原始编码，跳过")
            continue
        
        # 提取被合并的原始编码名称
        merged_list = [code.strip() for code in str(merged_codings).split(';') if code.strip()]
        
        if len(merged_list) < 2:
            print(f"编码 '{coding_name}': 被合并编码数量不足 ({len(merged_list)})，跳过计算")
            continue
        
        print(f"处理编码 '{coding_name}': {len(merged_list)} 个被合并编码")
        
        try:
            # 直接使用原始编码名称作为文本生成embedding
            embeddings = model.encode(merged_list, show_progress_bar=False, normalize_embeddings=True)
            
            # 计算信度指标
            reliability_metrics = calculate_reliability_metrics(embeddings)
            
            results.append({
                '编码类型': '开放式编码_被合并编码',
                '编码名称': coding_name,
                '被合并编码数量': reliability_metrics['item_count'],
                '克隆巴赫Alpha': reliability_metrics['cronbach_alpha'],
                '平均项目间相关': reliability_metrics['average_interitem_correlation'],
                '来源文件数量': row.get('来源文件数量', 'N/A'),
                '总出现次数': row.get('总出现次数', 'N/A'),
                '被合并编码示例': '; '.join(merged_list[:3]) + ('...' if len(merged_list) > 3 else '')
            })
            
            print(f"  → Alpha = {reliability_metrics['cronbach_alpha']:.4f}, 平均相关 = {reliability_metrics['average_interitem_correlation']:.4f}")
            
        except Exception as e:
            print(f"  → 处理编码 '{coding_name}' 时出错: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_axial_coding_reliability(model, axial_coding_df):
    """分析轴心编码的信度 - 基于被合并的原始编码"""
    print("\n开始分析轴心编码信度...")
    
    results = []
    
    for idx, row in axial_coding_df.iterrows():
        core_category = row['核心类别']
        merged_codings = row.get('被合并的原始编码', '')
        
        if pd.isna(merged_codings) or not merged_codings:
            print(f"核心类别 '{core_category}': 没有被合并的原始编码，跳过")
            continue
        
        # 提取被合并的原始编码名称
        merged_list = [code.strip() for code in str(merged_codings).split(';') if code.strip()]
        
        if len(merged_list) < 2:
            print(f"核心类别 '{core_category}': 被合并编码数量不足 ({len(merged_list)})，跳过计算")
            continue
        
        print(f"处理核心类别 '{core_category}': {len(merged_list)} 个被合并编码")
        
        try:
            # 直接使用原始编码名称作为文本生成embedding
            embeddings = model.encode(merged_list, show_progress_bar=False, normalize_embeddings=True)
            
            # 计算信度指标
            reliability_metrics = calculate_reliability_metrics(embeddings)
            
            results.append({
                '编码类型': '轴心编码',
                '核心类别': core_category.replace('**', '').strip(),
                '被合并编码数量': reliability_metrics['item_count'],
                '克隆巴赫Alpha': reliability_metrics['cronbach_alpha'],
                '平均项目间相关': reliability_metrics['average_interitem_correlation'],
                '来源文件数量': row.get('来源文件数量', 'N/A'),
                '总出现次数': row.get('总出现次数', 'N/A'),
                '被合并编码示例': '; '.join(merged_list[:3]) + ('...' if len(merged_list) > 3 else '')
            })
            
            print(f"  → Alpha = {reliability_metrics['cronbach_alpha']:.4f}, 平均相关 = {reliability_metrics['average_interitem_correlation']:.4f}")
            
        except Exception as e:
            print(f"  → 处理核心类别 '{core_category}' 时出错: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_selective_coding_reliability(model, axial_coding_df):
    """分析选择性编码的信度 - 基于核心类别名称"""
    print("\n开始分析选择性编码信度...")
    
    # 收集所有核心类别名称
    core_categories = []
    
    for idx, row in axial_coding_df.iterrows():
        core_category = row['核心类别'].replace('**', '').strip()
        if core_category and pd.notna(core_category):
            core_categories.append(core_category)
    
    # 计算核心类别之间的信度
    if len(core_categories) >= 2:
        print(f"选择性编码分析: {len(core_categories)} 个核心类别")
        print(f"核心类别列表: {core_categories}")
        
        try:
            # 直接使用核心类别名称生成embedding
            embeddings = model.encode(core_categories, show_progress_bar=False, normalize_embeddings=True)
            
            # 计算信度指标
            reliability_metrics = calculate_reliability_metrics(embeddings)
            
            # 计算类别间相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            # 创建结果
            results_data = [{
                '分析类型': '选择性编码总体信度',
                '核心类别数量': len(core_categories),
                '信度类型': '核心类别名称间信度',
                '克隆巴赫Alpha': reliability_metrics['cronbach_alpha'],
                '平均项目间相关': reliability_metrics['average_interitem_correlation']
            }]
            
            # 添加类别间相似度详情
            for i in range(len(core_categories)):
                for j in range(i + 1, len(core_categories)):
                    cat1 = core_categories[i]
                    cat2 = core_categories[j]
                    similarity = similarity_matrix[i, j]
                    
                    results_data.append({
                        '分析类型': f"类别间相似度 - {cat1} vs {cat2}",
                        '核心类别数量': 'N/A',
                        '信度类型': '两两相似度',
                        '克隆巴赫Alpha': similarity,
                        '平均项目间相关': similarity
                    })
            
            print(f"选择性编码总体信度: Alpha = {reliability_metrics['cronbach_alpha']:.4f}")
            return pd.DataFrame(results_data)
            
        except Exception as e:
            print(f"选择性编码分析出错: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"核心类别数量不足: {len(core_categories)}，需要至少2个类别")
    
    return pd.DataFrame()

def interpret_reliability_scores(alpha_score):
    """解释信度系数的意义"""
    if alpha_score >= 0.9:
        return "优秀"
    elif alpha_score >= 0.8:
        return "良好"
    elif alpha_score >= 0.7:
        return "可接受"
    elif alpha_score >= 0.6:
        return "一般"
    else:
        return "需要改进"

def main():
    """主函数"""
    # 加载模型
    model = load_embedding_model()
    
    try:
        # 读取数据
        print("读取数据文件...")
        open_coding_df = pd.read_csv(open_input_file, encoding='utf-8')
        axial_coding_df = pd.read_csv(axial_input_file, encoding='utf-8')
        
        print(f"开放式编码数据: {len(open_coding_df)} 行")
        print(f"轴心编码数据: {len(axial_coding_df)} 行")
        
        # 分析信度
        open_results_statements = analyze_open_coding_reliability_statements(model, open_coding_df)
        open_results_merged = analyze_open_coding_reliability_merged(model, open_coding_df)
        axial_results = analyze_axial_coding_reliability(model, axial_coding_df)
        selective_results = analyze_selective_coding_reliability(model, axial_coding_df)
        
        # 保存结果
        output_file = os.path.join(outpath, 'coding_reliability_analysis_comprehensive.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            if not open_results_statements.empty:
                open_results_statements.to_excel(writer, sheet_name='开放式编码_具体语句', index=False)
            
            if not open_results_merged.empty:
                open_results_merged.to_excel(writer, sheet_name='开放式编码_被合并编码', index=False)
            
            if not axial_results.empty:
                axial_results.to_excel(writer, sheet_name='轴心编码', index=False)
            
            if not selective_results.empty:
                selective_results.to_excel(writer, sheet_name='选择性编码', index=False)
        
        print(f"\n分析完成！结果已保存至: {output_file}")
        
        # 打印摘要统计
        print("\n=== 信度分析摘要 ===")
        
        if not open_results_statements.empty:
            open_avg_alpha = open_results_statements['克隆巴赫Alpha'].mean()
            open_best = open_results_statements.loc[open_results_statements['克隆巴赫Alpha'].idxmax()]
            open_worst = open_results_statements.loc[open_results_statements['克隆巴赫Alpha'].idxmin()]
            
            print(f"\n开放式编码信度（基于具体语句）:")
            print(f"  平均克隆巴赫Alpha: {open_avg_alpha:.4f} ({interpret_reliability_scores(open_avg_alpha)})")
            print(f"  最高信度: '{open_best['编码名称']}' = {open_best['克隆巴赫Alpha']:.4f}")
            print(f"  最低信度: '{open_worst['编码名称']}' = {open_worst['克隆巴赫Alpha']:.4f}")
            print(f"  分析编码数量: {len(open_results_statements)}")
        
        if not open_results_merged.empty:
            open_merged_avg_alpha = open_results_merged['克隆巴赫Alpha'].mean()
            open_merged_best = open_results_merged.loc[open_results_merged['克隆巴赫Alpha'].idxmax()]
            open_merged_worst = open_results_merged.loc[open_results_merged['克隆巴赫Alpha'].idxmin()]
            
            print(f"\n开放式编码信度（基于被合并编码）:")
            print(f"  平均克隆巴赫Alpha: {open_merged_avg_alpha:.4f} ({interpret_reliability_scores(open_merged_avg_alpha)})")
            print(f"  最高信度: '{open_merged_best['编码名称']}' = {open_merged_best['克隆巴赫Alpha']:.4f}")
            print(f"  最低信度: '{open_merged_worst['编码名称']}' = {open_merged_worst['克隆巴赫Alpha']:.4f}")
            print(f"  分析编码数量: {len(open_results_merged)}")
        
        if not axial_results.empty:
            axial_avg_alpha = axial_results['克隆巴赫Alpha'].mean()
            axial_best = axial_results.loc[axial_results['克隆巴赫Alpha'].idxmax()]
            axial_worst = axial_results.loc[axial_results['克隆巴赫Alpha'].idxmin()]
            
            print(f"\n轴心编码信度:")
            print(f"  平均克隆巴赫Alpha: {axial_avg_alpha:.4f} ({interpret_reliability_scores(axial_avg_alpha)})")
            print(f"  最高信度: '{axial_best['核心类别']}' = {axial_best['克隆巴赫Alpha']:.4f}")
            print(f"  最低信度: '{axial_worst['核心类别']}' = {axial_worst['克隆巴赫Alpha']:.4f}")
            print(f"  分析核心类别数量: {len(axial_results)}")
        
        if not selective_results.empty:
            overall_selective = selective_results[selective_results['分析类型'] == '选择性编码总体信度']
            if not overall_selective.empty:
                selective_alpha = overall_selective.iloc[0]['克隆巴赫Alpha']
                print(f"\n选择性编码信度:")
                print(f"  总体Alpha: {selective_alpha:.4f} ({interpret_reliability_scores(selective_alpha)})")
        
        print(f"\n信度解释标准:")
        print(f"  ≥ 0.9: 优秀")
        print(f"  0.8-0.9: 良好") 
        print(f"  0.7-0.8: 可接受")
        print(f"  0.6-0.7: 一般")
        print(f"  < 0.6: 需要改进")
    
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理内存
        del model
        gc.collect()

if __name__ == "__main__":
    main()