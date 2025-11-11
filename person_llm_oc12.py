# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:50:57 2025

@author: Fei Wang
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import jieba
import warnings
import datetime
import time
import gc
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# 配置路径
outpath = 'E:/Rdaima/RIS_theory/person_llm_oc12'
llm_models_dir = 'C:/workship/llm_models'

def load_embedding_model():
    """加载embedding模型"""
    model_file = os.path.join(llm_models_dir, "qwen3-embedding-0.6B")
    
    print(f"开始载入大语言模型文件 {model_file} ...")
    print("当前时间:", datetime.datetime.now())
    tbgn = time.time()
    model = SentenceTransformer(model_file)
    print(f"载入完毕，耗费时间：{(time.time() - tbgn) / 60:.2f} minutes")
    
    return model

def preprocess_text(text):
    """预处理文本：清洗、分词等"""
    if pd.isna(text):
        return ""
    
    # 转换为字符串并清洗
    text = str(text).strip()
    
    # 去除特殊字符和标点符号
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    
    # 使用jieba进行中文分词
    words = jieba.cut(text)
    
    # 过滤空字符并连接
    processed_text = ' '.join([word for word in words if word.strip()])
    
    return processed_text

def calculate_similarity_matrix_with_embedding(texts1, texts2, model):
    """使用大模型embedding计算两个文本列表之间的相似度矩阵"""
    print("正在使用大模型生成文本向量...")
    
    # 预处理文本
    processed_texts1 = [preprocess_text(text) for text in texts1]
    processed_texts2 = [preprocess_text(text) for text in texts2]
    
    # 过滤空文本
    valid_indices1 = [i for i, text in enumerate(processed_texts1) if text.strip()]
    valid_indices2 = [i for i, text in enumerate(processed_texts2) if text.strip()]
    
    valid_texts1 = [processed_texts1[i] for i in valid_indices1]
    valid_texts2 = [processed_texts2[i] for i in valid_indices2]
    
    if not valid_texts1 or not valid_texts2:
        print("警告: 有效文本为空，无法计算相似度")
        n = len(texts1)
        m = len(texts2)
        return np.zeros((n, m))
    
    # 使用大模型生成embedding
    print(f"正在为 {len(valid_texts1)} 个文本1和 {len(valid_texts2)} 个文本2生成embedding...")
    tbgn = time.time()
    
    embeddings1 = model.encode(valid_texts1, show_progress_bar=True, batch_size=32)
    embeddings2 = model.encode(valid_texts2, show_progress_bar=True, batch_size=32)
    
    print(f"Embedding生成完毕，耗费时间：{(time.time() - tbgn):.2f} 秒")
    
    # 计算余弦相似度
    print("正在计算相似度矩阵...")
    similarity_matrix_valid = cosine_similarity(embeddings1, embeddings2)
    
    # 创建完整的相似度矩阵（包含空文本）
    n = len(texts1)
    m = len(texts2)
    full_similarity_matrix = np.zeros((n, m))
    
    for i, idx1 in enumerate(valid_indices1):
        for j, idx2 in enumerate(valid_indices2):
            full_similarity_matrix[idx1, idx2] = similarity_matrix_valid[i, j]
    
    return full_similarity_matrix

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

def find_all_unique_pairings(texts1, texts2, model):
    """找到所有唯一的配对"""
    print("正在预处理文本...")
    
    print("正在计算相似度矩阵（使用大模型embedding）...")
    similarity_matrix = calculate_similarity_matrix_with_embedding(texts1, texts2, model)
    
    print("正在寻找最优配对...")
    pairings = find_optimal_pairings(similarity_matrix)
    
    return pairings, similarity_matrix

def calculate_overall_similarity_with_embedding(texts1, texts2, model):
    """使用大模型embedding计算两组文本的总体相似度"""
    processed_texts1 = [preprocess_text(text) for text in texts1]
    processed_texts2 = [preprocess_text(text) for text in texts2]
    
    # 将每组文本合并成一个长文本
    combined_text1 = ' '.join(processed_texts1)
    combined_text2 = ' '.join(processed_texts2)
    
    if not combined_text1.strip() or not combined_text2.strip():
        return 0.0
    
    # 使用大模型计算总体相似度
    print("正在计算总体相似度（使用大模型embedding）...")
    embeddings = model.encode([combined_text1, combined_text2])
    overall_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return overall_similarity

def load_data_files():
    """从文件加载数据"""
    # 文件路径 - 请根据实际情况修改
    file1_path = "E:/Rdaima/RIS_theory/person_oc12.xlsx"  # Excel文件
    file2_path = "E:/Rdaima/RIS_theory/embedding_second/Second_merge_open_coding.csv"  # CSV文件
    
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
        return None, None
    
    # 读取CSV文件
    try:
        # 尝试不同的编码方式
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
            return None, None
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None, None
    
    return df1, df2

def extract_coding_names(df1, df2):
    """从数据框中提取编码名称"""
    # 查找包含'编码'或'名称'的列
    coding_col1 = None
    coding_col2 = None
    
    # 在第一个数据框中查找
    for col in df1.columns:
        if '编码' in col or '名称' in col:
            coding_col1 = col
            break
    
    # 在第二个数据框中查找
    for col in df2.columns:
        if '编码' in col or '名称' in col:
            coding_col2 = col
            break
    
    # 如果没找到，使用第一列
    if coding_col1 is None and len(df1.columns) > 0:
        coding_col1 = df1.columns[0]
        print(f"文件1未找到编码名称列，使用第一列: {coding_col1}")
    
    if coding_col2 is None and len(df2.columns) > 0:
        coding_col2 = df2.columns[0]
        print(f"文件2未找到编码名称列，使用第一列: {coding_col2}")
    
    if coding_col1 is None or coding_col2 is None:
        print("无法找到合适的编码名称列")
        return None, None, None, None
    
    # 提取编码名称
    codes1 = df1[coding_col1].dropna().tolist()
    codes2 = df2[coding_col2].dropna().tolist()
    
    print(f"文件1编码列: {coding_col1}, 数量: {len(codes1)}")
    print(f"文件2编码列: {coding_col2}, 数量: {len(codes2)}")
    
    return codes1, codes2, coding_col1, coding_col2

def main():
    """主函数"""
    print("开始编码名称相似度分析（使用大模型embedding）")
    print("=" * 50)
    
    # 加载embedding模型
    model = load_embedding_model()
    
    # 加载数据文件
    df1, df2 = load_data_files()
    if df1 is None or df2 is None:
        print("数据加载失败，程序退出")
        return
    
    # 提取编码名称
    codes1, codes2, col1, col2 = extract_coding_names(df1, df2)
    if codes1 is None or codes2 is None:
        print("编码名称提取失败，程序退出")
        return
    
    # 显示编码示例
    print(f"\n文件1中的编码数量: {len(codes1)}")
    print(f"文件2中的编码数量: {len(codes2)}")
    print("\n文件1中的编码名称示例:")
    for i, code in enumerate(codes1[:5]):
        print(f"  {i+1}. {code}")
    
    print("\n文件2中的编码名称示例:")
    for i, code in enumerate(codes2[:5]):
        print(f"  {i+1}. {code}")
    
    # 找到所有唯一配对
    print("\n正在计算配对相似度（使用大模型embedding）...")
    pairings, similarity_matrix = find_all_unique_pairings(codes1, codes2, model)
    
    # 输出配对结果
    print(f"\n找到 {len(pairings)} 个唯一配对:")
    print("=" * 80)
    
    pairing_results = []
    for i, (idx1, idx2, similarity) in enumerate(pairings):
        code1 = codes1[idx1]
        code2 = codes2[idx2]
        pairing_results.append({
            '配对编号': i+1,
            '文件1编码': code1,
            '文件2编码': code2,
            '相似度': similarity
        })
        print(f"配对 {i+1}:")
        print(f"  文件1: {code1}")
        print(f"  文件2: {code2}")
        print(f"  相似度: {similarity:.4f}")
        print("-" * 60)
    
    # 计算平均相似度
    if pairings:
        avg_similarity = np.mean([sim for _, _, sim in pairings])
        print(f"\n平均相似度: {avg_similarity:.4f}")
    else:
        avg_similarity = 0
        print("\n未找到任何配对")
    
    # 计算总体相似度
    print("\n正在计算总体相似度（使用大模型embedding）...")
    overall_similarity = calculate_overall_similarity_with_embedding(codes1, codes2, model)
    print(f"总体相似度: {overall_similarity:.4f}")
    
    # 保存结果到Excel文件
    output_df = pd.DataFrame(pairing_results)
    output_file = os.path.join(outpath, "编码配对相似度分析结果.xlsx")
    
    # 创建输出目录
    os.makedirs(outpath, exist_ok=True)
    
    # 创建总结信息
    summary_data = {
        '统计项': ['文件1编码数量', '文件2编码数量', '配对数量', '平均相似度', '总体相似度'],
        '数值': [len(codes1), len(codes2), len(pairings), avg_similarity, overall_similarity]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # 保存相似度矩阵
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=[f"{col1}_{i}: {codes1[i]}" for i in range(len(codes1))],
        columns=[f"{col2}_{i}: {codes2[i]}" for i in range(len(codes2))]
    )
    
    # 保存到Excel，包含多个sheet
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='配对详情', index=False)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
            similarity_df.to_excel(writer, sheet_name='相似度矩阵')
        
        print(f"\n结果已保存到文件: {output_file}")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
        # 尝试保存为CSV
        try:
            output_df.to_csv(os.path.join(outpath, "编码配对相似度分析结果.csv"), index=False, encoding='utf-8-sig')
            summary_df.to_csv(os.path.join(outpath, "统计摘要.csv"), index=False, encoding='utf-8-sig')
            print("结果已保存为CSV文件")
        except Exception as e2:
            print(f"保存CSV文件也失败: {e2}")
    
    # 输出相似度最高的前10个配对
    if pairings:
        print(f"\n相似度最高的前10个配对:")
        print("=" * 80)
        sorted_pairings = sorted(pairings, key=lambda x: x[2], reverse=True)
        for i, (idx1, idx2, similarity) in enumerate(sorted_pairings[:10]):
            code1 = codes1[idx1]
            code2 = codes2[idx2]
            print(f"第{i+1}名 - 相似度: {similarity:.4f}")
            print(f"  文件1: {code1}")
            print(f"  文件2: {code2}")
            print("-" * 60)
    
    # 清理内存
    del model
    gc.collect()
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()