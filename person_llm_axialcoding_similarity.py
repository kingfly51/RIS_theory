# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 21:32:02 2025

@author: Fei Wang
"""
import os
import pandas as pd
import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc
from sentence_transformers import SentenceTransformer

# 路径配置
outpath = 'E:/Rdaima/RIS_theory/person_llm_axialcoding_similarity'
llm_models_dir = 'C:/workship/llm_models'

def save_text_file(file_path, content):
    """保存文本文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_and_process_data():
    """从两个文件加载数据并处理"""
    # 文件路径
    file1_path = "E:/Rdaima/RIS_theory/person_axial_coding.xlsx"
    file2_path = "E:/Rdaima/RIS_theory/Axial_output/轴心编码结果.csv"
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        raise FileNotFoundError(f"文件1不存在: {file1_path}")
    if not os.path.exists(file2_path):
        raise FileNotFoundError(f"文件2不存在: {file2_path}")
    
    print("正在读取文件...")
    
    # 读取Excel文件
    try:
        df1 = pd.read_excel(file1_path)
        print(f"成功读取Excel文件: {file1_path}")
        print(f"Excel文件列名: {df1.columns.tolist()}")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return None, None, None, None
    
    # 读取CSV文件
    try:
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        df2 = None
        for encoding in encodings:
            try:
                df2 = pd.read_csv(file2_path, encoding=encoding)
                print(f"成功读取CSV文件，编码: {encoding}")
                print(f"CSV文件列名: {df2.columns.tolist()}")
                break
            except UnicodeDecodeError:
                continue
        if df2 is None:
            print("无法读取CSV文件，尝试所有编码都失败")
            return None, None, None, None
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None, None, None, None
    
    # 提取并合并核心类别和类别定义
    combined_texts1, combined_texts2, col_info1, col_info2 = extract_and_combine_texts(df1, df2)
    
    return combined_texts1, combined_texts2, col_info1, col_info2

def extract_and_combine_texts(df1, df2):
    """提取并合并核心类别和类别定义"""
    # 查找文件1中的核心类别和类别定义列
    core_col1 = None
    definition_col1 = None
    
    for col in df1.columns:
        if '核心类别' in col:
            core_col1 = col
        elif '类别定义' in col or '定义' in col:
            definition_col1 = col
    
    if core_col1 is None and len(df1.columns) > 0:
        core_col1 = df1.columns[0]
        print(f"警告: 文件1中未找到'核心类别'列，使用第一列: {core_col1}")
    
    if definition_col1 is None and len(df1.columns) > 1:
        definition_col1 = df1.columns[1]
        print(f"警告: 文件1中未找到'类别定义'列，使用第二列: {definition_col1}")
    elif definition_col1 is None:
        definition_col1 = core_col1
        print(f"警告: 文件1中未找到'类别定义'列，使用核心类别列")
    
    # 查找文件2中的核心类别和类别定义列
    core_col2 = None
    definition_col2 = None
    
    for col in df2.columns:
        if '核心类别' in col:
            core_col2 = col
        elif '类别定义' in col or '定义' in col:
            definition_col2 = col
    
    if core_col2 is None and len(df2.columns) > 0:
        core_col2 = df2.columns[0]
        print(f"警告: 文件2中未找到'核心类别'列，使用第一列: {core_col2}")
    
    if definition_col2 is None and len(df2.columns) > 1:
        definition_col2 = df2.columns[1]
        print(f"警告: 文件2中未找到'类别定义'列，使用第二列: {definition_col2}")
    elif definition_col2 is None:
        definition_col2 = core_col2
        print(f"警告: 文件2中未找到'类别定义'列，使用核心类别列")
    
    print(f"文件1 - 核心类别列: {core_col1}, 类别定义列: {definition_col1}")
    print(f"文件2 - 核心类别列: {core_col2}, 类别定义列: {definition_col2}")
    
    # 提取并合并文本
    combined_texts1 = []
    for idx, row in df1.iterrows():
        core_text = str(row[core_col1]) if pd.notna(row[core_col1]) else ""
        definition_text = str(row[definition_col1]) if definition_col1 != core_col1 and pd.notna(row[definition_col1]) else ""
        
        # 合并核心类别和类别定义
        if definition_text and definition_text != core_text:
            combined_text = f"{core_text}：{definition_text}"
        else:
            combined_text = core_text
        
        combined_texts1.append(combined_text)
    
    combined_texts2 = []
    for idx, row in df2.iterrows():
        core_text = str(row[core_col2]) if pd.notna(row[core_col2]) else ""
        # 去除文件2核心类别中的**符号
        core_text = core_text.replace('**', '')
        definition_text = str(row[definition_col2]) if definition_col2 != core_col2 and pd.notna(row[definition_col2]) else ""
        
        # 合并核心类别和类别定义
        if definition_text and definition_text != core_text:
            combined_text = f"{core_text}：{definition_text}"
        else:
            combined_text = core_text
        
        combined_texts2.append(combined_text)
    
    print(f"文件1合并后文本数量: {len(combined_texts1)}")
    print(f"文件2合并后文本数量: {len(combined_texts2)}")
    
    # 返回列信息用于记录
    col_info1 = {'core_col': core_col1, 'definition_col': definition_col1}
    col_info2 = {'core_col': core_col2, 'definition_col': definition_col2}
    
    return combined_texts1, combined_texts2, col_info1, col_info2

def calculate_similarity_matrix(model, texts1, texts2):
    """计算两个文本列表之间的相似度矩阵"""
    print("正在生成文本embedding...")
    
    # 合并所有文本
    all_texts = texts1 + texts2
    
    # 批量生成embedding
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"已处理 {min(i + batch_size, len(all_texts))}/{len(all_texts)} 个文本")
    
    all_embeddings = np.array(all_embeddings)
    
    # 分割embedding矩阵
    embeddings1 = all_embeddings[:len(texts1)]
    embeddings2 = all_embeddings[len(texts1):]
    
    print("正在计算余弦相似度...")
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    
    return similarity_matrix

def find_optimal_pairings(similarity_matrix):
    """使用贪心算法找到最优配对"""
    n, m = similarity_matrix.shape
    paired_indices1 = set()
    paired_indices2 = set()
    pairings = []
    
    # 创建相似度列表并按相似度降序排序
    similarities = []
    for i in range(n):
        for j in range(m):
            similarities.append((i, j, similarity_matrix[i, j]))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # 贪心匹配
    for i, j, sim in similarities:
        if i not in paired_indices1 and j not in paired_indices2:
            pairings.append((i, j, sim))
            paired_indices1.add(i)
            paired_indices2.add(j)
            
            # 如果已经匹配完所有可能的对，就停止
            if len(paired_indices1) == min(n, m):
                break
    
    return pairings

def calculate_overall_similarity(model, texts1, texts2):
    """计算两组文本的总体相似度"""
    # 将每组文本合并成一个长文本
    combined_text1 = ' '.join(texts1)
    combined_text2 = ' '.join(texts2)
    
    if not combined_text1.strip() or not combined_text2.strip():
        return 0.0
    
    print("正在计算总体相似度...")
    embeddings = model.encode([combined_text1, combined_text2])
    overall_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return overall_similarity

def analyze_similarity_distribution(pairings, threshold=0.6):
    """分析相似度分布"""
    if not pairings:
        return {
            'total_pairs': 0,
            'pairs_above_threshold': 0,
            'percentage_above_threshold': 0,
            'mean_similarity': 0,
            'median_similarity': 0,
            'max_similarity': 0,
            'min_similarity': 0,
            'std_similarity': 0
        }
    
    similarities = [sim for _, _, sim in pairings]
    
    # 统计高于阈值的个体
    above_threshold = [sim for sim in similarities if sim > threshold]
    count_above = len(above_threshold)
    total_pairs = len(similarities)
    percentage = (count_above / total_pairs) * 100 if total_pairs > 0 else 0
    
    # 计算统计信息
    stats = {
        'total_pairs': total_pairs,
        'pairs_above_threshold': count_above,
        'percentage_above_threshold': percentage,
        'mean_similarity': np.mean(similarities),
        'median_similarity': np.median(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'std_similarity': np.std(similarities)
    }
    
    return stats

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(outpath, exist_ok=True)
    
    # 加载数据
    print("开始加载数据...")
    combined_texts1, combined_texts2, col_info1, col_info2 = load_and_process_data()
    
    if combined_texts1 is None or combined_texts2 is None:
        print("数据加载失败，程序退出")
        exit(1)
    
    print(f"文件1合并后文本数量: {len(combined_texts1)}")
    print(f"文件2合并后文本数量: {len(combined_texts2)}")
    
    # 显示合并后的文本示例
    print("\n文件1合并后文本示例:")
    for i, text in enumerate(combined_texts1[:5]):
        print(f"  {i+1}. {text}")
    
    print("\n文件2合并后文本示例:")
    for i, text in enumerate(combined_texts2[:5]):
        print(f"  {i+1}. {text}")
    
    # 加载模型
    model_file = os.path.join(llm_models_dir, "qwen3-embedding-0.6B")
    
    print(f"\n开始载入大语言模型文件 {model_file} ...")
    print("Now：", datetime.datetime.now())
    tbgn = time.time()
    model = SentenceTransformer(model_file)
    print("载入完毕，耗费时间：", (time.time() - tbgn) / 60, "minutes")
    
    # 1. 计算相似度矩阵和最优配对
    print("\n开始计算相似度矩阵...")
    tbgn = time.time()
    similarity_matrix = calculate_similarity_matrix(model, combined_texts1, combined_texts2)
    print(f"相似度矩阵计算完毕，耗费时间：{(time.time() - tbgn):.2f} 秒")
    
    print("\n正在寻找最优配对...")
    tbgn = time.time()
    pairings = find_optimal_pairings(similarity_matrix)
    print(f"最优配对寻找完毕，耗费时间：{(time.time() - tbgn):.2f} 秒")
    
    # 2. 计算总体相似度
    print("\n开始计算总体相似度...")
    tbgn = time.time()
    overall_similarity = calculate_overall_similarity(model, combined_texts1, combined_texts2)
    print(f"总体相似度计算完毕，耗费时间：{(time.time() - tbgn):.2f} 秒")
    
    # 分析相似度分布
    threshold = 0.6
    pair_stats = analyze_similarity_distribution(pairings, threshold)
    
    # 输出结果
    print(f"\n相似度分析结果 (阈值 = {threshold}):")
    print(f"总配对数量: {pair_stats['total_pairs']}")
    print(f"相似度 > {threshold} 的配对数量: {pair_stats['pairs_above_threshold']}")
    print(f"占比: {pair_stats['percentage_above_threshold']:.2f}%")
    print(f"平均相似度: {pair_stats['mean_similarity']:.4f}")
    print(f"中位数相似度: {pair_stats['median_similarity']:.4f}")
    print(f"最大相似度: {pair_stats['max_similarity']:.4f}")
    print(f"最小相似度: {pair_stats['min_similarity']:.4f}")
    print(f"两组总文本相似度: {overall_similarity:.4f}")
    
    # 保存配对结果
    pairing_results = []
    for i, (idx1, idx2, similarity) in enumerate(pairings):
        text1 = combined_texts1[idx1]
        text2 = combined_texts2[idx2]
        
        pairing_results.append({
            '配对编号': i + 1,
            '文件1合并文本': text1,
            '文件2合并文本': text2,
            '相似度': similarity
        })
    
    pairing_df = pd.DataFrame(pairing_results)
    pairing_file = os.path.join(outpath, '编码名称配对相似度结果.csv')
    pairing_df.to_csv(pairing_file, index=False, encoding='utf-8-sig')
    print(f"\n配对相似度结果保存至：{pairing_file}")
    
    # 保存相似度矩阵
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=[f"文件1_{i}" for i in range(len(combined_texts1))],
        columns=[f"文件2_{i}" for i in range(len(combined_texts2))]
    )
    matrix_file = os.path.join(outpath, '相似度矩阵.csv')
    similarity_df.to_csv(matrix_file, encoding='utf-8-sig')
    print(f"相似度矩阵保存至：{matrix_file}")
    
    # 保存总体统计结果
    overall_stats = {
        '文件1文本数量': len(combined_texts1),
        '文件2文本数量': len(combined_texts2),
        '总配对数量': pair_stats['total_pairs'],
        '相似度阈值': threshold,
        '高于阈值的配对数量': pair_stats['pairs_above_threshold'],
        '高于阈值占比(%)': pair_stats['percentage_above_threshold'],
        '配对平均相似度': pair_stats['mean_similarity'],
        '两组总文本相似度': overall_similarity,
        '最大相似度': pair_stats['max_similarity'],
        '最小相似度': pair_stats['min_similarity'],
        '文件1核心类别列': col_info1['core_col'],
        '文件1类别定义列': col_info1['definition_col'],
        '文件2核心类别列': col_info2['core_col'],
        '文件2类别定义列': col_info2['definition_col']
    }
    
    stats_df = pd.DataFrame([overall_stats])
    stats_file = os.path.join(outpath, '总体相似度统计.csv')
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"总体统计结果保存至：{stats_file}")
    
    # 清理内存
    del model
    gc.collect()
    
    print('\n工作完成!')
    print(f"共处理了 {len(combined_texts1)} 个文件1合并文本和 {len(combined_texts2)} 个文件2合并文本")
    print(f"生成 {len(pairings)} 个最优配对")
    print(f"相似度 > {threshold} 的配对占比: {pair_stats['percentage_above_threshold']:.2f}%")
    print(f"两组总文本相似度: {overall_similarity:.4f}")