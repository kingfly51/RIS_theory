# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:00:21 2025

@author: Feoi Wang
"""
import os
import sys
import pandas as pd
import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc
from sentence_transformers import SentenceTransformer

outpath = 'E:/Rdaima/RIS_theory/person_llm_opencoding_similarity'
# 本地化大语言模型数据
llm_models_dir = 'C:/workship/llm_models'

def save_text_file(file_path, content):
    """保存文本文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_and_process_pairs(folder1, folder2):
    """从两个文件夹中加载对应的文件并分别处理"""
    pair_data = []
    all_text1 = []  # 存储第一组所有文本
    all_text2 = []  # 存储第二组所有文本
    
    # 获取第一个文件夹中的所有CSV文件
    for filename in os.listdir(folder1):
        if filename.endswith('.csv') and not filename.endswith('_编码结果.csv'):
            base_name = filename.replace('.csv', '')
            
            # 构建两个文件的完整路径
            file1_path = os.path.join(folder1, filename)
            file2_name = f"{base_name}_编码结果.csv"
            file2_path = os.path.join(folder2, file2_name)
            
            # 检查第二个文件是否存在
            if os.path.exists(file2_path):
                try:
                    # 读取第一个文件（sub001.csv）
                    df1 = pd.read_csv(file1_path, encoding='utf-8')
                    # 读取第二个文件（sub001_编码结果.csv）
                    df2 = pd.read_csv(file2_path, encoding='utf-8')
                    
                    # 处理第一个文件的开放式编码列
                    if '开放式编码' in df1.columns:
                        text1 = ' '.join([str(x) for x in df1['开放式编码'].dropna() if str(x).strip() != ''])
                        all_text1.append(text1)  # 添加到第一组总文本
                    else:
                        text1 = ''
                        print(f"警告: {file1_path} 中没有'开放式编码'列")
                    
                    # 处理第二个文件的编码名称列
                    if '编码名称' in df2.columns:
                        text2 = ' '.join([str(x) for x in df2['编码名称'].dropna() if str(x).strip() != ''])
                        all_text2.append(text2)  # 添加到第二组总文本
                    else:
                        text2 = ''
                        print(f"警告: {file2_path} 中没有'编码名称'列")
                    
                    pair_data.append({
                        'file_base': base_name,
                        'file1': filename,
                        'file2': file2_name,
                        'text1': text1,
                        'text2': text2,
                        'text1_length': len(text1),
                        'text2_length': len(text2)
                    })
                    
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
            else:
                print(f"未找到对应文件: {file2_path}")
    
    # 拼接两组的总文本
    total_text1 = ' '.join(all_text1)
    total_text2 = ' '.join(all_text2)
    
    return pd.DataFrame(pair_data), total_text1, total_text2

def calculate_pair_similarities(pair_data, model):
    """计算每对文件之间的相似度"""
    similarities = []
    
    for _, pair in pair_data.iterrows():
        if pair['text1'] and pair['text2']:  # 确保两个文本都不为空
            # 对两个文本分别进行向量化
            embeddings = model.encode([pair['text1'], pair['text2']])
            # 计算相似度
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            similarities.append({
                'file_base': pair['file_base'],
                'file1': pair['file1'],
                'file2': pair['file2'],
                'text1_length': pair['text1_length'],
                'text2_length': pair['text2_length'],
                'similarity': similarity,
                'text1_preview': pair['text1'][:100] + '...' if len(pair['text1']) > 100 else pair['text1'],
                'text2_preview': pair['text2'][:100] + '...' if len(pair['text2']) > 100 else pair['text2']
            })
        else:
            print(f"跳过 {pair['file_base']}，因为其中一个文本为空")
    
    return pd.DataFrame(similarities)

def analyze_similarity_distribution(similarities_df, threshold=0.6):
    """分析相似度分布"""
    similarities = similarities_df['similarity'].values
    
    # 统计高于阈值的个体
    above_threshold = similarities[similarities > threshold]
    count_above = len(above_threshold)
    total_pairs = len(similarities)
    percentage = (count_above / total_pairs) * 100 if total_pairs > 0 else 0
    
    # 计算统计信息
    stats = {
        'total_pairs': total_pairs,
        'similarity_threshold': threshold,
        'pairs_above_threshold': count_above,
        'percentage_above_threshold': percentage,
        'mean_similarity': np.mean(similarities),
        'median_similarity': np.median(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'std_similarity': np.std(similarities)
    }
    
    return stats

def calculate_group_similarity(total_text1, total_text2, model):
    """计算两组总文本的相似度"""
    if total_text1 and total_text2:
        embeddings = model.encode([total_text1, total_text2])
        group_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return group_similarity
    else:
        return 0.0

if __name__ == '__main__':
    # 设置两个文件夹路径
    folder1 = 'E:/Rdaima/RIS_theory/person_output_csv_files'  # 包含 sub001.csv 等文件的文件夹
    folder2 = 'E:/Rdaima/RIS_theory/Open_output/individual_results'  # 包含 sub001_编码结果.csv 等文件的文件夹
    
    # 确保输出目录存在
    os.makedirs(outpath, exist_ok=True)
    
    # 加载文件对数据
    print("开始加载文件对数据...")
    pair_data, total_text1, total_text2 = load_and_process_pairs(folder1, folder2)
    
    if pair_data.empty:
        print("未找到匹配的文件对，请检查文件夹路径和文件命名")
        sys.exit(1)
    
    print(f"共找到 {len(pair_data)} 对文件")
    print(f"第一组总文本长度: {len(total_text1)}")
    print(f"第二组总文本长度: {len(total_text2)}")
    
    # 加载模型
    model_file = os.path.join(llm_models_dir, "qwen3-embedding-0.6B")
    
    print(f"开始载入大语言模型文件 {model_file} ...")
    print("Now：", datetime.datetime.now())
    tbgn = time.time()
    model = SentenceTransformer(model_file)
    print("载入完毕，耗费时间：", (time.time() - tbgn) / 60, "minutes")
    
    # 1. 计算每对文件之间的相似度
    print("\n开始计算每对文件之间的相似度...")
    tbgn = time.time()
    pair_similarities = calculate_pair_similarities(pair_data, model)
    print(f"文件对相似度计算完毕，耗费时间：{(time.time() - tbgn):.2f} 秒")
    
    # 保存文件对相似度结果
    pair_similarities_file = os.path.join(outpath, '文件对相似度结果.csv')
    pair_similarities.to_csv(pair_similarities_file, index=False, encoding='utf-8-sig')
    print(f"文件对相似度结果保存至：{pair_similarities_file}")
    
    # 分析文件对相似度分布
    threshold = 0.6
    pair_stats = analyze_similarity_distribution(pair_similarities, threshold)
    
    print(f"\n文件对相似度分析结果 (阈值 = {threshold}):")
    print(f"总文件对数: {pair_stats['total_pairs']}")
    print(f"相似度 > {threshold} 的文件对数: {pair_stats['pairs_above_threshold']}")
    print(f"占比: {pair_stats['percentage_above_threshold']:.2f}%")
    print(f"平均相似度: {pair_stats['mean_similarity']:.4f}")
    print(f"中位数相似度: {pair_stats['median_similarity']:.4f}")
    print(f"最大相似度: {pair_stats['max_similarity']:.4f}")
    print(f"最小相似度: {pair_stats['min_similarity']:.4f}")
    
    # 2. 计算两组总文本的相似度
    print("\n开始计算两组总文本的相似度...")
    tbgn = time.time()
    group_similarity = calculate_group_similarity(total_text1, total_text2, model)
    print(f"两组总文本相似度计算完毕，耗费时间：{(time.time() - tbgn):.2f} 秒")
    
    print(f"\n两组总文本相似度分析结果:")
    print(f"第一组总文本长度: {len(total_text1)}")
    print(f"第二组总文本长度: {len(total_text2)}")
    print(f"两组总文本相似度: {group_similarity:.4f}")
    
    # 保存总体统计结果
    overall_stats = {
        '总文件对数': pair_stats['total_pairs'],
        '相似度阈值': threshold,
        '高于阈值的文件对数': pair_stats['pairs_above_threshold'],
        '高于阈值占比(%)': pair_stats['percentage_above_threshold'],
        '文件对平均相似度': pair_stats['mean_similarity'],
        '两组总文本相似度': group_similarity,
        '第一组总文本长度': len(total_text1),
        '第二组总文本长度': len(total_text2)
    }
    
    stats_df = pd.DataFrame([overall_stats])
    stats_file = os.path.join(outpath, '总体相似度统计.csv')
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"总体统计结果保存至：{stats_file}")
    
    # 保存相似度最高的前10对文件
    top_similarities = pair_similarities.nlargest(10, 'similarity')
    top_similarities_file = os.path.join(outpath, '相似度最高的前10对文件.csv')
    top_similarities.to_csv(top_similarities_file, index=False, encoding='utf-8-sig')
    print(f"相似度最高的前10对文件保存至：{top_similarities_file}")
    
    # 保存相似度最低的前10对文件
    bottom_similarities = pair_similarities.nsmallest(10, 'similarity')
    bottom_similarities_file = os.path.join(outpath, '相似度最低的前10对文件.csv')
    bottom_similarities.to_csv(bottom_similarities_file, index=False, encoding='utf-8-sig')
    print(f"相似度最低的前10对文件保存至：{bottom_similarities_file}")
    
    # 清理内存
    del model
    gc.collect()
    
    print('\n工作完成!')
    print(f"共处理了 {len(pair_data)} 对文件")
    print(f"相似度 > {threshold} 的文件对占比: {pair_stats['percentage_above_threshold']:.2f}%")
    print(f"两组总文本相似度: {group_similarity:.4f}")