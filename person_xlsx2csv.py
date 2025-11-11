# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:46:35 2025

@author: Fei Wang
"""

import pandas as pd
import os

# 读取 person.xlsx 文件
df = pd.read_excel('E:/Rdaima/RIS_theory/person_open_coding.xlsx')

# 确保输出目录存在
output_dir = 'E:/Rdaima/RIS_theory/person_output_csv_files'
os.makedirs(output_dir, exist_ok=True)

# 按“文件名称”分组
grouped = df.groupby('文件名称')

for filename, group in grouped:
    # 提取 sub00k 部分
    base_name = filename.replace('.txt', '')
    
    # 构造输出CSV文件名
    output_csv_path = os.path.join(output_dir, f'{base_name}.csv')
    
    # 构造与“编码1.csv”相同结构的DataFrame
    output_df = pd.DataFrame({
        '文件名称': group['文件名称'],
        '原始文本': group['原始文本'],
        '开放式编码': group['开放式编码'],
        '编码来源语句': group['编码来源语句'],
        '编码定义': group['编码定义']
    })
    
    # 保存为CSV
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f'已生成: {output_csv_path}')

print("所有文件生成完成！")