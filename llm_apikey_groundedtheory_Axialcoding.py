# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:43:29 2025

@author: dong
"""
import os
import time
import pandas as pd
from openai import OpenAI

# 配置信息 - 修改为使用第二次合并后的结果
INPUT_FILE = 'E:/Rdaima/RIS_theory/embedding_second/第二次合并后的编码结果.csv'  # 第二次合并后的编码结果文件
OUTPUT_DIR1 = 'E:/Rdaima/RIS_theory/Axial_output2'
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

3.您已完成对参与者关于"您认为是什么原因造成了休息羞耻呢？"这一问题的访谈文本的初始开放式编码以及对部分相似编码的初步合并。
'''

task2 = '''
当前正在进行关于"休息羞耻"成因的扎根理论研究，现已完成开放式编码阶段，进入轴心编码阶段。您的任务是基于背景知识中休息羞耻的定义和开放式编码结果，进行基于扎根理论的轴心编码。

###核心要求
成因导向：重点分析与导致"休息羞耻"直接或间接相关的驱动性因素、背景和环境。
层次化：将分散的开放式编码组织为有层次结构的类别体系，体现概念层级。
机制化：揭示不同因素如何相互作用导致休息羞耻，阐明作用路径。
数据驱动：所有类别关系必须基于开放式编码的文本证据支持。
完整性：构建的理论框架必须全面覆盖所有开放式编码。

### 轴心编码流程：
第一步，明确核心现象：确定"休息羞耻"作为整个分析过程围绕的轴心。
第二步，系统分析所有开放式编码，识别编码间的概念相似性、语义关联性和逻辑关系。
第三步，基于编码间的内在联系，将具有共同属性或指向同一现象的开放式编码进行聚类。
第四步，对每个编码簇进行概念化提炼，形成核心范畴并赋予理论性命名。
第五步，鉴于本研究聚焦于成因，请重点运用编码范式中与"前因"相关的部分，识别各核心范畴在导致"休息羞耻"中的理论角色：
  因果条件：引发、导致或促成"休息羞耻"的核心驱动因素。
  情境脉络：构成休息羞耻发生背景的环境性因素。
  干预条件：调节因果条件作用强度的因素，它可能加剧或缓解休息羞耻的产生。
第六步，明确每个核心范畴的理论定义，阐述其在休息羞耻产生过程中的具体作用。
第七步，验证编码归类的合理性，确保：
  每个核心范畴包含≥2个开放式编码
  所有开放式编码都被合理分配
  范畴定义清晰且具有理论解释力

###输出格式
请以表格形式呈现轴心编码结果，表格包含以下五列：
| 核心类别 | 包含的开放式编码 | 类别定义 | 范式角色 | 作用机制描述 |

各列具体要求：
核心类别：识别出的高层次概念，每个范畴包含≥2个开放式编码
包含的开放式编码：列出该类别下的所有编码名称（用逗号分隔）
类别定义：阐述该核心类别的理论含义和在成因中的作用
范式角色：从 "因果条件/情境脉络/干预条件" 中选择（三选一）
作用机制描述：具体说明该范畴如何作为前因，影响或导致"休息羞耻"的产生。

### 关键注意事项
1. 禁止直接使用休息羞耻的四个维度作为核心类别名称
2. 每个核心类别必须包含至少2个开放式编码
3. 所有分析应紧紧围绕"成因"这一焦点。
4. 确保所有开放式编码都被分配且仅分配到一个核心类别
5. 基于编码定义的内在联系进行合理聚类
6. 包含的开放式编码总数应与输入编码数量一致
7. 本次编码的目标是构建一个关于"休息羞耻"成因的理论模型

请直接输出最终的轴心编码表格，不输出思考过程或其他任何内容。表格应包含完整的核心类别体系，全面反映休息羞耻的多元成因机制。
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
        
        # 确保有5列且不是表头行（通过检查是否包含中文或数字来判断）
        if len(cells) >= 5:
            first_cell = cells[0]
            # 简单判断：如果第一个单元格不包含常见表头词，且包含中文字符，则认为是数据行
            if ('核心类别' not in first_cell and 
                '包含的开放式编码' not in first_cell and
                any('\u4e00' <= char <= '\u9fff' for char in first_cell)):
                rows.append({
                    '核心类别': cells[0],
                    '包含的开放式编码': cells[1],
                    '类别定义': cells[2],
                    '范式角色': cells[3],
                    '作用机制描述': cells[4]
                })
    
    return rows

def prepare_open_coding_data(df):
    """准备开放式编码数据供轴心编码使用 - 只保留编码名称和编码定义"""
    coding_data = []
    
    for _, row in df.iterrows():
        coding_info = {
            '编码名称': row['编码名称'],
            '编码定义': row['编码定义']
        }
        coding_data.append(coding_info)
    
    return coding_data

def format_coding_data_for_prompt(coding_data):
    """将开放式编码数据格式化为适合prompt的文本 - 只包含编码名称和定义"""
    formatted_text = "开放式编码数据汇总：\n\n"
    
    for i, code in enumerate(coding_data, 1):
        formatted_text += f"{i}. 编码名称: {code['编码名称']}\n"
        formatted_text += f"   编码定义: {code['编码定义']}\n\n"
    
    formatted_text += f"总计: {len(coding_data)} 个开放式编码\n"
    return formatted_text

def merge_axial_coding_data(axial_coding_data, original_df):
    """
    基于轴心编码结果合并原始数据中的其他字段
    
    参数:
    axial_coding_data: 大模型返回的轴心编码结果
    original_df: 原始的开放式编码DataFrame
    
    返回:
    合并后的轴心编码数据
    """
    merged_data = []
    
    for axial_row in axial_coding_data:
        # 获取该核心类别包含的所有编码名称
        coding_names_str = axial_row['包含的开放式编码']
        
        # 解析编码名称（可能用逗号、分号等分隔）
        coding_names = []
        for separator in ['，', '、', ',', ';', '；']:
            if separator in coding_names_str:
                coding_names = [name.strip() for name in coding_names_str.split(separator)]
                break
        else:
            # 如果没有找到分隔符，尝试按空格分割
            coding_names = [name.strip() for name in coding_names_str.split()]
        
        # 在原始数据中查找这些编码
        matched_codes = original_df[original_df['编码名称'].isin(coding_names)]
        
        if len(matched_codes) == 0:
            print(f"警告: 未找到编码名称: {coding_names}")
            # 如果完全匹配失败，尝试部分匹配
            matched_indices = []
            for coding_name in coding_names:
                mask = original_df['编码名称'].str.contains(coding_name, na=False)
                if mask.any():
                    matched_indices.extend(original_df[mask].index.tolist())
            
            if matched_indices:
                matched_codes = original_df.loc[matched_indices]
        
        if len(matched_codes) > 0:
            # 计算合并后的统计信息
            source_files = set()
            total_occurrences = 0
            all_statements = []
            all_involved_files = set()
            all_merged_codes = set()
            
            for _, code_row in matched_codes.iterrows():
                # 处理来源文件数量
                if pd.notna(code_row.get('来源文件数量')):
                    source_files_count = code_row['来源文件数量']
                    # 如果是数值，直接累加；如果是字符串，需要解析
                    if isinstance(source_files_count, (int, float)):
                        source_files.add(str(source_files_count))
                    else:
                        source_files.add(str(source_files_count))
                
                # 累加总出现次数
                total_occurrences += code_row['总出现次数']
                
                # 收集具体语句汇总
                if pd.notna(code_row['具体语句汇总']):
                    all_statements.append(str(code_row['具体语句汇总']))
                
                # 处理涉及文件
                if pd.notna(code_row['涉及文件']):
                    files = str(code_row['涉及文件']).split(',')
                    all_involved_files.update([f.strip() for f in files])
                
                # 处理被合并的原始编码
                if pd.notna(code_row['被合并的原始编码']):
                    merged = str(code_row['被合并的原始编码']).split(';')
                    all_merged_codes.update([m.strip() for m in merged])
            
            # 创建合并后的行
            merged_row = axial_row.copy()
            # 添加本地计算的统计字段
            merged_row['来源文件数量'] = len(all_involved_files)  # 使用涉及文件的去重数量
            merged_row['总出现次数'] = total_occurrences
            merged_row['具体语句汇总'] = ' | '.join(all_statements)
            merged_row['涉及文件'] = ', '.join(sorted(all_involved_files))
            merged_row['被合并的原始编码'] = '; '.join(sorted(all_merged_codes))
            
            merged_data.append(merged_row)
        else:
            print(f"严重警告: 完全无法匹配编码名称: {coding_names}")
            # 保留原始的大模型输出，但添加空的统计字段
            merged_row = axial_row.copy()
            merged_row['来源文件数量'] = 0
            merged_row['总出现次数'] = 0
            merged_row['具体语句汇总'] = ''
            merged_row['涉及文件'] = ''
            merged_row['被合并的原始编码'] = ''
            merged_data.append(merged_row)
    
    return merged_data

def calculate_axial_statistics(merged_axial_data):
    """计算轴心编码的统计信息"""
    print("\n=== 轴心编码统计信息 ===")
    
    # 计算总文件数量和出现次数
    total_files = sum(int(row.get('来源文件数量', 0)) for row in merged_axial_data)
    total_occurrences = sum(int(row.get('总出现次数', 0)) for row in merged_axial_data)
    
    print(f"总来源文件数量: {total_files}")
    print(f"总出现次数: {total_occurrences}")
    
    # 按核心类别统计
    core_categories = {}
    for row in merged_axial_data:
        core_cat = row['核心类别']
        if core_cat not in core_categories:
            core_categories[core_cat] = {
                'categories': [],
                'total_files': 0,
                'total_occurrences': 0
            }
        core_categories[core_cat]['categories'].append(row)
        core_categories[core_cat]['total_files'] += int(row.get('来源文件数量', 0))
        core_categories[core_cat]['total_occurrences'] += int(row.get('总出现次数', 0))
    
    print(f"\n核心类别分布:")
    for core_cat, stats in core_categories.items():
        print(f"- {core_cat}: {len(stats['categories'])}个类别, {stats['total_files']}个来源文件, {stats['total_occurrences']}次出现")
    
    # 按范式角色统计
    paradigm_roles = {}
    for row in merged_axial_data:
        role = row.get('范式角色', '未知')
        if role not in paradigm_roles:
            paradigm_roles[role] = 0
        paradigm_roles[role] += 1
    
    print(f"\n范式角色分布:")
    for role, count in paradigm_roles.items():
        print(f"- {role}: {count}个类别")
    
    return total_files, total_occurrences

if __name__ == '__main__':
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR1):
        os.makedirs(OUTPUT_DIR1)
    
    # 读取第二次合并后的编码结果
    try:
        df_open_coding = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
        print(f"成功读取第二次合并后的编码结果，共 {len(df_open_coding)} 个编码")
        
        # 保存完整的原始数据用于后续合并
        original_df_full = df_open_coding.copy()
        
        # 准备数据（只包含编码名称和定义）
        coding_data = prepare_open_coding_data(df_open_coding)
        print(f"用于轴心编码的编码数量: {len(coding_data)}")
        
        if len(coding_data) == 0:
            print("错误: 没有可用的编码数据，请检查数据文件")
            exit(1)
        
        formatted_coding_data = format_coding_data_for_prompt(coding_data)
        
        # 初始化API客户端
        access_key_id = os.getenv('QWEN_API_KEY')
        
        if not access_key_id:
            print("错误: 未找到QWEN_API_KEY环境变量")
            exit(1)
            
        client = OpenAI(
            api_key=access_key_id,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 构建系统指令
        instruction = f'>>> 背景知识\n""""""{bk_info}""""""\n\n>>> 任务\n{task2}'
        
        # 完整的用户输入 - 只传递编码名称和定义
        user_content = f"{formatted_coding_data}\n\n请基于以上开放式编码数据进行轴心编码分析。"
        
        print("正在进行轴心编码分析...")
        print(f"输入编码数量: {len(coding_data)}")
        
        # API调用
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.5,
            top_p=0.5,
        )
        
        msg = response.choices[0].message.content
        
        print("收到模型响应，正在解析...")
        print("=" * 50)
        print(msg)
        print("=" * 50)
        
        # 解析轴心编码结果
        axial_coding_data = parse_table_response(msg)
        
        if axial_coding_data:
            print(f"解析到 {len(axial_coding_data)} 个轴心编码类别")
            
            # 合并原始数据中的其他字段
            print("正在合并原始数据字段...")
            merged_axial_data = merge_axial_coding_data(axial_coding_data, original_df_full)
            
            # 定义完整的输出列顺序
            output_columns = [
                '核心类别', 
                '包含的开放式编码', 
                '类别定义', 
                '范式角色', 
                '作用机制描述',
                '来源文件数量',
                '总出现次数',
                '具体语句汇总',
                '涉及文件',
                '被合并的原始编码'
            ]
            
            # 保存轴心编码结果
            df_axial = pd.DataFrame(merged_axial_data)
            # 确保所有列都存在
            for col in output_columns:
                if col not in df_axial.columns:
                    df_axial[col] = ""
            
            df_axial = df_axial[output_columns]
            axial_output_path = os.path.join(OUTPUT_DIR1, '轴心编码结果.csv')
            df_axial.to_csv(axial_output_path, index=False, encoding='utf-8-sig')
            
            # 保存原始响应
            raw_response_path = os.path.join(OUTPUT_DIR1, '轴心编码原始响应.txt')
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                f.write(msg)
            
            print(f"\n轴心编码完成!")
            print(f"生成的核心类别数量: {len(set([row['核心类别'] for row in merged_axial_data]))}")
            print(f"生成的类别数量: {len(merged_axial_data)}")
            print(f"轴心编码结果已保存至: {axial_output_path}")
            
            # 计算并显示详细统计信息
            total_files, total_occurrences = calculate_axial_statistics(merged_axial_data)
            
        else:
            print("警告: 未解析到有效的轴心编码结果")
            # 保存原始响应以便调试
            with open(os.path.join(OUTPUT_DIR1, '轴心编码调试响应.txt'), 'w', encoding='utf-8') as f:
                f.write(msg)
                
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}")
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print('\n轴心编码工作完成!')