# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:51:55 2025

@author: Fei Wang
"""

import os
import pandas as pd
from openai import OpenAI

# 配置信息
AXIAL_FILE = 'E:/Rdaima/RIS_theory/Axial_output2/轴心编码结果.csv'  # 轴心编码结果文件
OUTPUT_DIR = 'E:/Rdaima/RIS_theory/Selective_output3'
MODEL = "qwen3-max"

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

3.您已完成对参与者关于"您认为是什么原因造成了休息羞耻呢？"这一问题的访谈文本的开放式编码和轴心编码工作，
轴心编码结果已形成完整的核心类别体系。
'''

task3 = '''
当前正在进行关于"休息羞耻"成因的扎根理论研究，现已完成开放式编码和轴心编码阶段，进入选择性编码阶段。

### 核心任务
基于前两个阶段的编码结果，进行选择性编码，以构建一个关于"休息羞耻"成因的整合性理论框架。整个分析必须始终围绕"休息羞耻"这一核心现象，并聚焦于解释其成因。

### 核心要求
1.  核心范畴统领：从轴心编码结果中选择一个最具解释力的核心范畴。该范畴必须是能够作为理论核心，系统性地整合其他所有范畴，并对"休息羞耻"的成因提供根本性解释。
2.  成因导向的范式分析：运用编码范式中与成因相关的部分，系统化地组织理论框架。重点关注导致和影响"休息羞耻"的条件因素。
3.  过程化叙事：围绕核心范畴发展出逻辑连贯的"故事线"，清晰地阐述各种条件如何相互作用，动态地导致"休息羞耻"的产生。
4.  严格的数据扎根：所有理论建构必须严格基于前期编码结果，确保理论从数据中自然涌现。

### 操作流程
第一步：识别核心范畴
从轴心编码的范畴中，基于以下标准选择一个作为核心范畴：
    中心性：与大多数其他范畴存在密集且重要的关联。
    解释力：能够对"休息羞耻为何产生"提供最深刻、最根本的理论解释。
    频现性：在数据中反复出现且意义显著。
    统领性：能够作为理论"骨架"，自然地整合其他范畴。

第二步：构建以核心范畴为中心的成因网络
将其他范畴与核心范畴联系起来，明确它们在解释"休息羞耻"成因中的具体角色。请重点识别和阐述以下关系：
    因果条件：哪些范畴是导致"休息羞耻"的直接驱动因素？
    情境脉络：哪些范畴构成了"休息羞耻"发生的背景环境？
    干预条件：哪些范畴调节（加剧或缓解）了"休息羞耻"的形成过程？

第三步：撰写理论故事线
用连贯的叙述性文字，讲述一个关于"休息羞耻"如何形成的理论故事。这个故事应以核心范畴为主线，有机地整合第二步中识别的各类条件因素，展现其动态的相互作用过程。

第四步：理论验证与总结
    验证所有重要的轴心编码范畴都在理论框架中得到了合理解释。
    为最终构建的理论命名，并总结其核心观点。

### 输出格式
请输出以下四个部分，不包含任何思考过程：

#### 一、核心范畴
    名称：[核心范畴的理论名称]
    定义与解释：[阐述其理论内涵，并说明它为何及如何能成为核心范畴]

#### 二、休息羞耻的成因理论框架
    因果条件：[列出相关范畴，并描述其作为驱动因素的作用机制]
    情境脉络：[列出相关范畴，并描述其作为背景环境的作用]
    干预条件：[列出相关范畴，并描述其如何调节成因过程]

#### 三、理论故事线
[在此处撰写一段连贯、逻辑清晰的理论叙述，阐明休息羞耻的成因过程]

#### 四、理论总结
    理论名称：[反映理论核心的命名]
    核心理论命题：[用一句精炼的话概括理论的核心观点]
    理论贡献：[简要说明该框架对理解休息羞耻成因的价值]

### 关键注意事项
1.核心现象固定：牢记核心现象是"休息羞耻"，核心范畴是用于解释它的、从数据中浮现的理论概念。
2.聚焦成因：分析的重点是"休息羞耻"为何及如何产生，无需牵强地寻找和描述其"行动策略"与"后果"。
3.选择而非创造：核心范畴必须从已有的轴心编码结果中依据标准选择，而非凭空创造。
4.关系导向：确保清晰地描述范畴之间的作用机制，而非仅仅罗列范畴。
'''

def format_axial_data_for_prompt(df_axial):
    """将轴心编码数据格式化为适合prompt的文本 - 包含所有必要信息"""
    formatted_text = "轴心编码结果汇总：\n\n"
    
    # 按核心类别分组显示
    core_categories = df_axial['核心类别'].unique()
    
    for core_cat in core_categories:
        formatted_text += f"【{core_cat}】\n"
        
        # 获取该核心类别下的所有类别
        categories_in_core = df_axial[df_axial['核心类别'] == core_cat]
        
        for i, row in enumerate(categories_in_core.itertuples(), 1):
            formatted_text += f"  {i}. 类别名称: {getattr(row, '核心类别', '')}\n"
            formatted_text += f"     包含编码: {getattr(row, '包含的开放式编码', '')}\n"
            formatted_text += f"     类别定义: {getattr(row, '类别定义', '')}\n"
            formatted_text += f"     范式角色: {getattr(row, '范式角色', '')}\n"
            formatted_text += f"     作用机制: {getattr(row, '作用机制描述', '')}\n"
            
            # 添加统计信息
            source_files = getattr(row, '来源文件数量', 0)
            total_occurrences = getattr(row, '总出现次数', 0)
            formatted_text += f"     统计信息: {source_files}个文件, {total_occurrences}次出现\n\n"
    
    # 添加总体统计摘要
    total_categories = len(df_axial)
    total_core_categories = len(core_categories)
    total_files = df_axial['来源文件数量'].sum()
    total_occurrences = df_axial['总出现次数'].sum()
    
    formatted_text += f"统计摘要：\n"
    formatted_text += f"- 核心类别数量: {total_core_categories}\n"
    formatted_text += f"- 轴心类别总数: {total_categories}\n"
    formatted_text += f"- 总来源文件数量: {total_files}\n"
    formatted_text += f"- 总出现次数: {total_occurrences}\n\n"
    
    return formatted_text

def prepare_axial_data(df_axial):
    """准备轴心编码数据 - 包含所有列用于选择性编码分析"""
    # 选择性编码需要所有信息来进行综合分析
    required_columns = [
        '核心类别', 
        '包含的开放式编码', 
        '类别定义', 
        '范式角色', 
        '作用机制描述',
        '来源文件数量',
        '总出现次数'
    ]
    
    # 检查是否存在所有必需的列
    missing_columns = [col for col in required_columns if col not in df_axial.columns]
    if missing_columns:
        print(f"警告: 缺少以下列: {missing_columns}")
        # 使用可用的列
        available_columns = [col for col in required_columns if col in df_axial.columns]
        df_filtered = df_axial[available_columns].copy()
    else:
        df_filtered = df_axial[required_columns].copy()
    
    return df_filtered

def calculate_axial_statistics(df_axial):
    """计算轴心编码的统计信息"""
    print("\n=== 轴心编码输入统计 ===")
    
    total_categories = len(df_axial)
    core_categories = df_axial['核心类别'].unique() if '核心类别' in df_axial.columns else []
    total_core_categories = len(core_categories)
    
    total_files = df_axial['来源文件数量'].sum() if '来源文件数量' in df_axial.columns else 0
    total_occurrences = df_axial['总出现次数'].sum() if '总出现次数' in df_axial.columns else 0
    
    print(f"核心类别数量: {total_core_categories}")
    print(f"轴心类别总数: {total_categories}")
    print(f"总来源文件数量: {total_files}")
    print(f"总出现次数: {total_occurrences}")
    
    if len(core_categories) > 0:
        print(f"\n核心类别列表:")
        for core_cat in core_categories:
            categories_in_core = df_axial[df_axial['核心类别'] == core_cat]
            core_files = categories_in_core['来源文件数量'].sum() if '来源文件数量' in categories_in_core.columns else 0
            core_occurrences = categories_in_core['总出现次数'].sum() if '总出现次数' in categories_in_core.columns else 0
            print(f"- {core_cat}: {len(categories_in_core)}个类别, {core_files}个文件, {core_occurrences}次出现")
    
    # 按范式角色统计
    if '范式角色' in df_axial.columns:
        paradigm_roles = df_axial['范式角色'].value_counts()
        print(f"\n范式角色分布:")
        for role, count in paradigm_roles.items():
            print(f"- {role}: {count}个类别")
    
    return total_core_categories, total_categories, total_files, total_occurrences

def save_selective_coding_results(response_text, output_dir):
    """保存选择性编码结果"""
    # 保存原始响应
    raw_response_path = os.path.join(output_dir, '选择性编码原始响应.txt')
    with open(raw_response_path, 'w', encoding='utf-8') as f:
        f.write(response_text)
    
    # 尝试解析并保存结构化结果
    try:
        # 解析各部分内容
        lines = response_text.split('\n')
        core_category_section = []
        framework_section = []
        story_line_section = []
        theory_summary_section = []
        
        current_section = None
        for line in lines:
            if '一、核心范畴' in line or '#### 一、核心范畴' in line:
                current_section = 'core'
                continue
            elif '二、休息羞耻的成因理论框架' in line or '#### 二、休息羞耻的成因理论框架' in line:
                current_section = 'framework'
                continue
            elif '三、理论故事线' in line or '#### 三、理论故事线' in line:
                current_section = 'story'
                continue
            elif '四、理论总结' in line or '#### 四、理论总结' in line:
                current_section = 'theory'
                continue
            
            if current_section == 'core' and line.strip():
                core_category_section.append(line.strip())
            elif current_section == 'framework' and line.strip():
                framework_section.append(line.strip())
            elif current_section == 'story' and line.strip():
                story_line_section.append(line.strip())
            elif current_section == 'theory' and line.strip():
                theory_summary_section.append(line.strip())
        
        # 保存解析后的各部分
        with open(os.path.join(output_dir, '核心范畴分析.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(core_category_section))
        
        with open(os.path.join(output_dir, '成因理论框架.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(framework_section))
        
        with open(os.path.join(output_dir, '理论故事线.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(story_line_section))
        
        with open(os.path.join(output_dir, '理论总结.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(theory_summary_section))
            
        print(f"选择性编码结构化结果已保存至: {output_dir}")
        
    except Exception as e:
        print(f"解析结构化结果时出错: {e}")

if __name__ == '__main__':
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 读取轴心编码结果
    try:
        df_axial = pd.read_csv(AXIAL_FILE, encoding='utf-8-sig')
        print(f"成功读取轴心编码结果，共 {len(df_axial)} 个类别")
        
        # 准备轴心编码数据 - 包含所有必要信息
        df_axial_filtered = prepare_axial_data(df_axial)
        print(f"用于选择性编码的列: {list(df_axial_filtered.columns)}")
        
        # 计算并显示输入统计信息
        core_cat_count, total_cat_count, total_files, total_occurrences = calculate_axial_statistics(df_axial_filtered)
        
        # 准备轴心编码数据供提示词使用
        formatted_axial_data = format_axial_data_for_prompt(df_axial_filtered)
        
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
        instruction = f'>>> 背景知识\n""""""{bk_info}""""""\n\n>>> 任务\n{task3}'
        
        # 完整的用户输入 - 包含完整的轴心编码信息
        user_content = f"{formatted_axial_data}\n\n请基于以上轴心编码结果进行选择性编码分析，构建整合性理论框架。"
        
        print("\n正在进行选择性编码分析...")
        print(f"输入数据: {core_cat_count}个核心类别, {total_cat_count}个轴心类别")
        
        # API调用
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,  # 降低温度以提高一致性
            top_p=0.5,
        )
        
        msg = response.choices[0].message.content
        
        print("收到模型响应，正在保存结果...")
        
        # 打印响应内容
        print("\n" + "="*50)
        print("选择性编码结果:")
        print("="*50)
        print(msg)
        
        # 保存选择性编码结果
        save_selective_coding_results(msg, OUTPUT_DIR)
        
        print(f"\n选择性编码完成!")
        print(f"结果已保存至: {OUTPUT_DIR}")
                
    except FileNotFoundError:
        print(f"错误: 找不到文件 {AXIAL_FILE}")
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print('\n选择性编码工作完成!')