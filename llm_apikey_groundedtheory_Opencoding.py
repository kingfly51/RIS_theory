# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 11:07:13 2025

@author: Fei Wang
"""
# 本程序是通过API调用远程大模型，
# 实现对访谈文本的扎根理论的开放式编码工作。


import os
import time
import pandas as pd
from openai import OpenAI

# 配置信息
INPUT_DIR = 'E:/Rdaima/RIS_theory/reason_RIS_high'  # reason_RIS_high
OUTPUT_DIR = 'E:/Rdaima/RIS_theory/Open_output'
MODEL = "qwen3-max"  # 或其他支持的远程模型

# 背景知识和任务定义
bk_info = '''
1.您是位社会科学领域的学者，擅长运用访谈数据和扎根理论，并精通访谈文本编码。

2.休息不耐受，在中国文化环境下也被称之为休息羞耻，指的是个体在休息时因负面情绪（如焦虑、内疚、羞耻等）、
认知偏差（如对休息的条件化或污名化）、强迫性思维（如无法摆脱待办事务的压力）以及社会比较（如与他人竞争状态的对比）而产生的心理不适体验。
休息不耐受最核心的维度是负面情绪。它包含焦虑、内疚、羞耻等负面情绪，以及在休息过程中遭受各种负面评价的不良体验。
除此之外，还存在另外三个维度：社会比较、强迫性思维和认知偏差。认知偏差指在休息过程中产生的错误认知，
认为休息具有条件性且对休息的定义存在偏见。强迫性思维指的是在休息时，
人们会不由自主地回想起待办事项（如工作或任务）以及这些事务带来的压力。
社会比较指的是在休息时感知到竞争氛围，关注周围同伴是否也在休息，并将自己与他们进行比较。
这四个维度彼此的相互作用构成了休息不耐受。
'''

task1 = '''您的任务是根据{背景知识}中休息羞耻的定义，对参与者回答"您认为是什么原因造成了休息羞耻呢？"这一问题的文本进行基于扎根理论的开放式编码，
旨在提取造成休息羞耻的潜在原因。

### 核心要求：
开放性：完全从文本数据出发，不预设任何理论框架
具体性：每个编码必须基于文本中的具体陈述
概念化：将具体陈述提炼为概念性编码标签
穷尽性：尽可能识别文本中所有潜在的原因因素

### 严格禁止：
禁止直接使用"社会比较"、"强迫性思维"、"认知偏差"、"负面情绪"等维度标签作为编码名称
禁止将休息羞耻的表现、结果、定义编码为原因
禁止输出重复、语义重叠或高度相似的编码

### 编码流程：
第一步，逐句阅读文本，识别包含原因信息的陈述
第二步，为每个相关陈述创建初步编码
第三步，在编码过程中不断比较新编码与已有编码的相似性
第四步，将初步编码提炼为适当抽象层次的概念性编码名称
第五步，合并相同原因的陈述，确保每个独特原因只对应一个编码
第六步，确保每个最终编码都有明确、完整的文本证据支持

### 输出格式：
请以表格形式呈现编码结果，表格包含以下四列：
| 编码名称 | 编码定义 | 编码来源具体语句 | 对应源编码数量 |

### 编码质量标准：
每个编码对应一个独特的原因机制
编码必须基于文本中的具体陈述
编码名称应简洁明确，具有适当的概念层次
编码定义要清晰解释该因素如何导致休息羞耻
具体语句引用要完整且忠实于原文
源编码数量准确统计同一编码下的不同陈述条数

请直接输出最终的编码表格，不输出思考过程或其他任何内容。
'''

def parse_table_response(response_text):
    """解析模型返回的表格文本"""
    rows = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        # 跳过表头分隔线和非表格行
        if '----' in line or '|' not in line:
            continue
        
        # 清理表格格式
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        
        # 确保有4列且不是表头行（通过检查是否包含中文或数字来判断）
        if len(cells) >= 4:
            first_cell = cells[0]
            # 简单判断：如果第一个单元格不包含常见表头词，且包含中文字符，则认为是数据行
            if ('编码名称' not in first_cell and 
                any('\u4e00' <= char <= '\u9fff' for char in first_cell)):
                rows.append({
                    '编码名称': cells[0],
                    '编码定义': cells[1],
                    '编码来源具体语句': cells[2],
                    '对应源编码数量': cells[3]
                })
    
    return rows

def merge_coding_results(all_results):
    """整合所有编码结果"""
    merged_codes = {}
    
    for result in all_results:
        filename = result['filename']
        coding_data = result['coding_data']
        
        for code in coding_data:
            code_name = code['编码名称']
            
            if code_name not in merged_codes:
                # 新编码，创建记录
                merged_codes[code_name] = {
                    '编码名称': code_name,
                    '编码定义': code['编码定义'],
                    '来源文件': [],
                    '具体语句': [],
                    '总出现次数': 0,
                    '源文件数量': 0
                }
            
            # 更新编码信息
            merged_codes[code_name]['来源文件'].append(filename)
            merged_codes[code_name]['具体语句'].append(code['编码来源具体语句'])
            merged_codes[code_name]['总出现次数'] += int(code['对应源编码数量']) if code['对应源编码数量'].isdigit() else 1
            merged_codes[code_name]['源文件数量'] = len(set(merged_codes[code_name]['来源文件']))
    
    return merged_codes

def create_final_coding_table(merged_codes):
    """创建最终整合编码表"""
    final_data = []
    
    for code_name, code_info in merged_codes.items():
        # 合并所有具体语句
        all_statements = ' | '.join([f"[文件: {file}] {stmt}" 
                                   for file, stmt in zip(code_info['来源文件'], code_info['具体语句'])])
        
        final_data.append({
            '编码名称': code_name,
            '编码定义': code_info['编码定义'],
            '来源文件数量': code_info['源文件数量'],
            '总出现次数': code_info['总出现次数'],
            '具体语句汇总': all_statements,
            '涉及文件': ', '.join(code_info['来源文件'])
        })
    
    return pd.DataFrame(final_data)

def save_individual_results(filename, coding_data, individual_dir):
    """保存单个文件的编码结果"""
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    
    df_individual = pd.DataFrame(coding_data)
    output_path = os.path.join(individual_dir, f"{os.path.splitext(filename)[0]}_编码结果.csv")
    df_individual.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"已保存单个文件编码结果: {output_path}")

if __name__ == '__main__':
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 初始化API客户端
    access_key_id = os.getenv('QWEN_API_KEY')  # 从环境变量获取API密钥
    
    client = OpenAI(
        api_key=access_key_id,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 构建系统指令
    instruction = f'>>> 背景知识\n""""""{bk_info}""""""\n\n>>> 任务\n{task1}'

    all_coding_results = []
    
    # 处理每个文件
    input_files = os.listdir(INPUT_DIR)
    for input_file in input_files:
        if input_file.endswith('.txt'):  # 只处理文本文件
            try:
                with open(os.path.join(INPUT_DIR, input_file), 'r', encoding='utf-8') as f:
                    mytext = f.read(3000)  # 读取前3000字符
                
                print(f"正在处理文件: {input_file}")
                
                # API调用
                time.sleep(1)  # 限速，避免请求过快
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": f"文本 < {mytext} > no_think"}
                    ],
                    temperature=0.5,
                    top_p=0.5
                )
                msg = response.choices[0].message.content
                
                # 解析编码结果
                coding_data = parse_table_response(msg)
                
                if coding_data:
                    # 保存单个文件结果
                    individual_dir = os.path.join(OUTPUT_DIR, 'individual_results')
                    save_individual_results(input_file, coding_data, individual_dir)
                    
                    # 添加到总结果
                    all_coding_results.append({
                        'filename': input_file,
                        'coding_data': coding_data,
                        'raw_response': msg
                    })
                    
                    print(f"文件 {input_file} 提取到 {len(coding_data)} 个编码")
                else:
                    print(f"警告: 文件 {input_file} 未提取到有效编码")
                    # 保存原始响应以便调试
                    with open(os.path.join(OUTPUT_DIR, f"{input_file}_raw_response.txt"), 'w', encoding='utf-8') as f:
                        f.write(msg)
                    
            except Exception as e:
                print(f"处理文件 {input_file} 时出错: {e}")
    
    # 整合所有编码结果
    if all_coding_results:
        print("\n开始整合所有编码结果...")
        merged_codes = merge_coding_results(all_coding_results)
        final_df = create_final_coding_table(merged_codes)
        
        # 按出现频率排序
        final_df = final_df.sort_values(['来源文件数量', '总出现次数'], ascending=[False, False])
        
        # 保存整合结果
        final_output_path = os.path.join(OUTPUT_DIR, '整合编码结果.csv')
        final_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
        
        # 保存详细结果（包含原始响应）
        detailed_results = []
        for result in all_coding_results:
            for code in result['coding_data']:
                detailed_results.append({
                    '来源文件': result['filename'],
                    '编码名称': code['编码名称'],
                    '编码定义': code['编码定义'],
                    '具体语句': code['编码来源具体语句'],
                    '出现次数': code['对应源编码数量']
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_output_path = os.path.join(OUTPUT_DIR, '详细编码结果.csv')
        detailed_df.to_csv(detailed_output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n整合完成!")
        print(f"总共处理文件数: {len(all_coding_results)}")
        print(f"提取的唯一编码数: {len(merged_codes)}")
        print(f"整合结果已保存至: {final_output_path}")
        print(f"详细结果已保存至: {detailed_output_path}")
        
        # 打印统计信息
        print(f"\n编码统计:")
        print(f"最多出现的编码: {final_df.iloc[0]['编码名称']} (出现在 {final_df.iloc[0]['来源文件数量']} 个文件中)")
        print(f"平均每个文件的编码数: {sum(len(result['coding_data']) for result in all_coding_results) / len(all_coding_results):.1f}")
        
    else:
        print("未提取到任何编码结果")
    
    print('\n工作完成!')
