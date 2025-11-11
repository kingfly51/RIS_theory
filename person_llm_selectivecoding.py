# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 19:30:51 2025

@author: Fei Wang
"""
import os
import os.path
import numpy as np
from sentence_transformers import SentenceTransformer
import datetime
import time
import gc

# 本地化大语言模型数据
llm_models_dir = 'C:/workship/llm_models'

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def read_text_file(file_path):
    """读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def calculate_files_similarity(model, file1_path, file2_path):
    """计算两个文本文件的相似度"""
    # 读取文件内容
    text1 = read_text_file(file1_path)
    text2 = read_text_file(file2_path)
    
    if text1 is None or text2 is None:
        return None
    
    print(f"文件1路径: {file1_path}")
    print(f"文件2路径: {file2_path}")
    print(f"文件1内容长度: {len(text1)} 字符")
    print(f"文件2内容长度: {len(text2)} 字符")
    
    # 显示部分内容预览
    preview_length = 100
    print(f"\n文件1内容预览: {text1[:preview_length]}...")
    print(f"文件2内容预览: {text2[:preview_length]}...")
    
    # 生成文本向量
    embeddings = model.encode([text1, text2])
    
    print(f'文本向量长度：{len(embeddings[0])}')
    
    # 计算相似度
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    
    return similarity

if __name__ == '__main__':
    # 模型文件路径
    model_file = os.path.join(llm_models_dir, "qwen3-embedding-0.6B")
    
    # 要比较的两个文本文件路径
    llm_file = 'E:/Rdaima/RIS_theory/Selective_output/理论故事线.txt'
    expert_file = 'E:/Rdaima/RIS_theory/person_selective_coding.txt'
    
    print(f"开始载入大语言模型文件 {model_file} ...")
    print("当前时间：", datetime.datetime.now())
    tbgn = time.time()
    
    # 载入模型
    model = SentenceTransformer(model_file)
    
    print("载入完毕，耗费时间：", (time.time() - tbgn) / 60, "minutes")
    
    # 计算文件相似度
    print("\n" + "="*50)
    print("开始计算两个文本文件的相似度")
    print("="*50)
    
    similarity = calculate_files_similarity(model, llm_file, expert_file)
    
    if similarity is not None:
        print(f"\n相似度计算结果：")
        print(f"两个文件的余弦相似度: {similarity:.4f}")
        
        # 添加相似度等级描述
        if similarity >= 0.8:
            similarity_level = "极高相似度"
        elif similarity >= 0.6:
            similarity_level = "高相似度"
        elif similarity >= 0.4:
            similarity_level = "中等相似度"
        elif similarity >= 0.2:
            similarity_level = "低相似度"
        else:
            similarity_level = "极低相似度"
            
        print(f"相似度等级: {similarity_level}")
    else:
        print("无法计算相似度，请检查文件路径和内容")
    
    # 清理资源
    del model
    gc.collect()  # 手动触发垃圾回收
    
    print('\n工作完成!')

