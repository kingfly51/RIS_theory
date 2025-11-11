# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 10:03:38 2025

@author: Fei Wang
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

# 自定义文件保存函数，替代 ccpl_lib
def save_file(file_path, content):
    """保存内容到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"文件已保存: {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def load_merged_coding_data(csv_file):
    """读取合并后的编码结果CSV文件"""
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 提取编码名称、定义和被合并的原始编码
    coding_names = df['编码名称'].tolist()
    coding_definitions = df['编码定义'].tolist()
    merged_original_codes = df['被合并的原始编码'].tolist()
    
    # 创建完整的文本用于向量化（名称+定义+被合并的原始编码）
    coding_texts = []
    for name, definition, merged_codes in zip(coding_names, coding_definitions, merged_original_codes):
        if pd.isna(merged_codes):
            full_text = f"{name}：{definition}"
        else:
            full_text = f"{name}：{definition} 被合并的编码：{merged_codes}"
        coding_texts.append(full_text)
    
    return coding_names, coding_definitions, merged_original_codes, coding_texts, df

def merge_similar_codes_second(similarity_matrix, coding_names, threshold=0.75):
    """第二次合并相似度高于阈值的编码"""
    n = len(coding_names)
    merged_indices = set()
    merged_groups = []
    
    for i in range(n):
        if i in merged_indices:
            continue
            
        # 找到与当前编码相似的所有编码
        similar_indices = [i]
        for j in range(i+1, n):
            if j not in merged_indices and similarity_matrix[i][j] > threshold:
                similar_indices.append(j)
                merged_indices.add(j)
        
        merged_indices.add(i)
        merged_groups.append(similar_indices)
    
    return merged_groups

def find_most_representative_code_second(group, similarity_matrix):
    """找到与组内其他编码总体相似性最高的编码（第二次合并）"""
    if len(group) == 1:
        return group[0]  # 单个编码直接返回
    
    best_index = None
    best_avg_similarity = -1
    
    for i in group:
        # 计算当前编码与组内其他编码的平均相似度
        similarities = []
        for j in group:
            if i != j:  # 排除与自身的比较
                similarities.append(similarity_matrix[i][j])
        
        if similarities:  # 确保列表不为空
            avg_similarity = np.mean(similarities)
            
            # 选择平均相似度最高的编码
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_index = i
    
    return best_index

def create_second_merged_dataframe(merged_groups, coding_names, coding_definitions, merged_original_codes, original_df, similarity_matrix):
    """创建第二次合并后的数据框"""
    merged_data = []
    
    for group in merged_groups:
        if len(group) == 1:
            # 单个编码，保持不变
            idx = group[0]
            # 计算单个编码的来源文件数量
            files_str = original_df.iloc[idx]['涉及文件']
            if pd.notna(files_str):
                source_file_count = len([f.strip() for f in files_str.split(',')])
            else:
                source_file_count = 0
                
            merged_data.append({
                '编码名称': coding_names[idx],
                '编码定义': coding_definitions[idx],
                '来源文件数量': source_file_count,
                '总出现次数': original_df.iloc[idx]['总出现次数'],
                '具体语句汇总': original_df.iloc[idx]['具体语句汇总'],
                '涉及文件': original_df.iloc[idx]['涉及文件'],
                '被合并的原始编码': merged_original_codes[idx],
                '第二次合并的编码': "无",  # 单个编码，没有第二次合并
                '代表性评分': 1.0  # 单个编码的代表性为1
            })
        else:
            # 合并多个编码
            merged_names = [coding_names[i] for i in group]
            merged_definitions = [coding_definitions[i] for i in group]
            merged_originals = [merged_original_codes[i] for i in group]
            
            # 选择与组内其他编码总体相似性最高的编码
            best_idx = find_most_representative_code_second(group, similarity_matrix)
            main_definition = coding_definitions[best_idx]
            main_name = coding_names[best_idx]
            
            # 计算代表性评分（平均相似度）
            similarities = []
            for j in group:
                if best_idx != j:
                    similarities.append(similarity_matrix[best_idx][j])
            representative_score = np.mean(similarities) if similarities else 1.0
            
            # 合并相关数据
            total_occurrence = sum(original_df.iloc[i]['总出现次数'] for i in group)
            
            # 计算合并后的来源文件数量（去重）
            all_files = set()
            for i in group:
                files_str = original_df.iloc[i]['涉及文件']
                if pd.notna(files_str):
                    files = [f.strip() for f in files_str.split(',')]
                    all_files.update(files)
            merged_files = ", ".join(sorted(all_files))
            source_file_count = len(all_files)  # 去重后的文件数量
            
            # 合并具体语句汇总（去重）
            all_statements = []
            for i in group:
                statements = original_df.iloc[i]['具体语句汇总']
                if pd.notna(statements):
                    all_statements.append(statements)
            merged_statements = " | ".join(all_statements)
            
            # 合并被合并的原始编码（第一次合并的历史）
            all_merged_codes = []
            for original_code_str in merged_originals:
                if pd.notna(original_code_str):
                    codes = [code.strip() for code in original_code_str.split(';')]
                    all_merged_codes.extend(codes)
            merged_codes_str = "; ".join(sorted(set(all_merged_codes)))
            
            # 记录第二次合并的编码（当前这次合并的编码）
            second_merged_codes = "; ".join(merged_names)
            
            merged_data.append({
                '编码名称': f"{main_name}",
                '编码定义': main_definition,
                '来源文件数量': source_file_count,
                '总出现次数': total_occurrence,
                '具体语句汇总': merged_statements,
                '涉及文件': merged_files,
                '被合并的原始编码': merged_codes_str,  # 包含第一次和第二次合并的所有原始编码
                '第二次合并的编码': second_merged_codes,  # 只记录第二次合并的编码
                '代表性评分': representative_score
            })
    
    return pd.DataFrame(merged_data)

def calculate_second_file_statistics(second_merged_df, original_df):
    """计算第二次合并的文件统计信息"""
    print("\n=== 第二次合并文件数量统计 ===")
    
    # 计算第二次合并后的总文件数量
    total_second_merged_files = second_merged_df['来源文件数量'].sum()
    print(f"第二次合并后总来源文件数量: {total_second_merged_files}")
    
    # 计算第一次合并数据的总文件数量
    first_merged_total_files = 0
    for idx in range(len(original_df)):
        files_str = original_df.iloc[idx]['涉及文件']
        if pd.notna(files_str):
            files_count = len([f.strip() for f in files_str.split(',')])
            first_merged_total_files += files_count
    print(f"第一次合并后总来源文件数量: {first_merged_total_files}")
    
    # 计算第二次合并后所有涉及的文件总数（去重）
    all_second_merged_files = set()
    for idx in range(len(second_merged_df)):
        files_str = second_merged_df.iloc[idx]['涉及文件']
        if pd.notna(files_str):
            files = [f.strip() for f in files_str.split(',')]
            all_second_merged_files.update(files)
    actual_second_merged_file_count = len(all_second_merged_files)
    print(f"第二次合并实际涉及的不同文件数量: {actual_second_merged_file_count}")
    
    if first_merged_total_files > 0:
        reduction_rate = ((first_merged_total_files - actual_second_merged_file_count) / first_merged_total_files * 100)
        print(f"第二次合并文件去重率: {reduction_rate:.2f}%")
    
    # 显示每个第二次合并编码的文件数量分布
    print("\n第二次合并各编码文件数量分布:")
    for index, row in second_merged_df.iterrows():
        second_merge_info = "（第二次合并）" if row['第二次合并的编码'] != "无" else ""
        print(f"  {row['编码名称']}{second_merge_info}: {row['来源文件数量']} 个文件")
    
    return total_second_merged_files

def main():
    # 设置路径
    outpath = 'E:/Rdaima/RIS_theory/embedding_second'
    llm_models_dir = 'C:/workship/llm_models'
    
    # 确保输出目录存在
    os.makedirs(outpath, exist_ok=True)
    
    # 设置输入文件路径（使用第一次合并后的结果）
    csv_file = 'E:/Rdaima/RIS_theory/embedding_first/合并后的编码结果.csv'
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件不存在 - {csv_file}")
        return
    
    # 加载合并后的编码数据
    print("开始加载合并后的编码数据...")
    coding_names, coding_definitions, merged_original_codes, coding_texts, original_df = load_merged_coding_data(csv_file)
    print(f"共加载 {len(coding_names)} 个编码")
    
    # 加载模型
    model_file = os.path.join(llm_models_dir, "qwen3-embedding-0.6B")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_file):
        print(f"警告: 模型文件不存在，使用默认模型 - {model_file}")
        model = SentenceTransformer('qwen3-embedding-0.6B')  # 使用轻量级默认模型
    else:
        print(f"开始载入大语言模型文件 {model_file} ...")
        print("Now：", datetime.datetime.now())
        tbgn = time.time()
        model = SentenceTransformer(model_file)
        print("载入完毕，耗费时间：", (time.time() - tbgn) / 60, "minutes")
    
    # 对编码文本进行向量化
    print("开始对编码定义进行向量化 ...")
    print("Now：", datetime.datetime.now())
    tbgn = time.time()

    embeddings = model.encode(coding_texts)
    print("向量化完毕，耗费时间：", (time.time() - tbgn) / 60, "minutes")
    
    # 保存向量化结果
    out_file = os.path.join(outpath, 'second_coding_embeddings.txt')
    buf_out = ''
    for i in range(len(coding_names)):
        vec = [coding_names[i]] + [str(x) for x in embeddings[i]]
        outstring = ','.join([str(istr) for istr in vec])
        buf_out += outstring + '\n'
    save_file(out_file, buf_out)
    print(f"第二次编码的向量化数据保存至：{out_file}")
    
    # 计算相似度矩阵
    print("计算编码相似度...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # 保存相似度结果
    out_file = os.path.join(outpath, 'second_coding_similarity_matrix.txt')
    similarity_output = '编码1,编码2,相似度\n'
    for i in range(len(coding_names)):
        for j in range(i+1, len(coding_names)):
            similarity_output += f'{coding_names[i]},{coding_names[j]},{similarity_matrix[i][j]:.4f}\n'
    save_file(out_file, similarity_output)
    print(f"第二次编码相似度矩阵保存至：{out_file}")
    
    # 第二次合并相似编码（相似度>0.75）
    print("开始第二次合并相似编码（阈值=0.75）...")
    merged_groups = merge_similar_codes_second(similarity_matrix, coding_names, threshold=0.75)
    
    # 创建第二次合并后的数据框
    second_merged_df = create_second_merged_dataframe(merged_groups, coding_names, coding_definitions, merged_original_codes, original_df, similarity_matrix)
    
    # 计算文件统计信息
    total_source_files = calculate_second_file_statistics(second_merged_df, original_df)
    
    # 保存第二次合并结果
    second_merged_file = os.path.join(outpath, '第二次合并后的编码结果.csv')
    second_merged_df.to_csv(second_merged_file, index=False, encoding='utf-8-sig')
    print(f"第二次合并后的编码结果保存至：{second_merged_file}")
    
    # 保存简化版（不包含合并历史列）
    columns_to_drop = ['被合并的原始编码', '第二次合并的编码', '代表性评分']
    simplified_df = second_merged_df.drop(columns=[col for col in columns_to_drop if col in second_merged_df.columns])
    simplified_file = os.path.join(outpath, 'Second_merge_open_coding.csv')
    simplified_df.to_csv(simplified_file, index=False, encoding='utf-8-sig')
    print(f"第二次简化版合并结果保存至：{simplified_file}")
    
    # 输出统计信息
    first_merged_count = len(coding_names)
    second_merged_count = len(second_merged_df)
    reduction = first_merged_count - second_merged_count
    reduction_rate = (reduction / first_merged_count) * 100
    
    print(f"\n=== 第二次合并统计信息 ===")
    print(f"第一次合并后编码数量：{first_merged_count}")
    print(f"第二次合并后编码数量：{second_merged_count}")
    print(f"减少编码数量：{reduction}")
    print(f"压缩率：{reduction_rate:.2f}%")
    print(f"第二次合并后总来源文件数量：{total_source_files}")
    
    # 显示被合并的组和代表性编码
    print(f"\n=== 第二次合并组详情 ===")
    multi_merge_count = 0
    for i, group in enumerate(merged_groups):
        if len(group) > 1:
            multi_merge_count += 1
            group_names = [coding_names[idx] for idx in group]
            best_idx = find_most_representative_code_second(group, similarity_matrix)
            best_name = coding_names[best_idx]
            
            # 计算代表性评分
            similarities = []
            for j in group:
                if best_idx != j:
                    similarities.append(similarity_matrix[best_idx][j])
            rep_score = np.mean(similarities) if similarities else 1.0
            
            # 获取该组合并后的文件数量
            merged_files_count = second_merged_df.iloc[i]['来源文件数量']
            original_files_count = sum(len([f.strip() for f in original_df.iloc[idx]['涉及文件'].split(',')]) 
                                     for idx in group if pd.notna(original_df.iloc[idx]['涉及文件']))
            
            print(f"组 {i+1}（{len(group)}个编码）: ")
            print(f"  代表性编码: {best_name} (评分: {rep_score:.4f})")
            print(f"  文件数量: {merged_files_count} (合并前: {original_files_count})")
            print(f"  被合并编码: {', '.join([name for name in group_names if name != best_name])}")
            print()
    
    print(f"\n共合并了 {multi_merge_count} 个编码组")
    
    # 清理内存
    del embeddings
    del model
    del similarity_matrix
    gc.collect()
    
    print('\n第二次合并工作完成!\n')

if __name__ == '__main__':
    main()