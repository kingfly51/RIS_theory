library(readxl)
A_68 <- read_excel("D:/Rdaima/RIS_theory/对理论专家评分/336618282_按文本_休息羞耻成因理论可靠性评分A_68.xlsx")
B_56 <- read_excel("D:/Rdaima/RIS_theory/对理论专家评分/336631132_按文本_休息羞耻成因理论可靠性评分B_56.xlsx")

# 加载必要的包
library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)

# 查看两个数据集的列名，确认expert和llms列
colnames(A_68)
colnames(B_56)

# 合并两个数据集
combined_data <- bind_rows(A_68, B_56)
writexl::write_xlsx(combined_data,"combined_data62.xlsx")
# 提取expert和llms的7个指标列
expert_cols <- grep("^expert_", colnames(combined_data), value = TRUE)
llms_cols <- grep("^llms_", colnames(combined_data), value = TRUE)

# 确保两个组的列是对应的（去掉前缀后名称相同）
expert_measures <- gsub("^expert_", "", expert_cols)
llms_measures <- gsub("^llms_", "", llms_cols)

# 检查指标是否一致
print("Expert measures:")
print(expert_measures)
print("LLMS measures:")
print(llms_measures)

# 重塑数据为长格式（配对版本）
expert_long <- combined_data %>%
  select(all_of(expert_cols)) %>%
  mutate(id = row_number(), group = "expert") %>%
  pivot_longer(
    cols = all_of(expert_cols),
    names_to = "measure",
    values_to = "score"
  ) %>%
  mutate(measure = gsub("^expert_", "", measure))

llms_long <- combined_data %>%
  select(all_of(llms_cols)) %>%
  mutate(id = row_number(), group = "llms") %>%  # 使用相同的id，保持配对关系
  pivot_longer(
    cols = all_of(llms_cols),
    names_to = "measure",
    values_to = "score"
  ) %>%
  mutate(measure = gsub("^llms_", "", measure))

# 合并长格式数据
long_data <- bind_rows(expert_long, llms_long)
long_data$score <- as.numeric(long_data$score)

# 进行配对样本t检验
paired_t_test_results <- long_data %>%
  group_by(measure) %>%
  summarise(
    t_statistic = t.test(score[group == "expert"], 
                         score[group == "llms"], 
                         paired = TRUE)$statistic,
    p_value = t.test(score[group == "expert"], 
                     score[group == "llms"], 
                     paired = TRUE)$p.value,
    df = t.test(score[group == "expert"], 
                score[group == "llms"], 
                paired = TRUE)$parameter,
    expert_mean = mean(score[group == "expert"], na.rm = TRUE),
    llms_mean = mean(score[group == "llms"], na.rm = TRUE),
    mean_difference = expert_mean - llms_mean,
    expert_sd = sd(score[group == "expert"], na.rm = TRUE),
    llms_sd = sd(score[group == "llms"], na.rm = TRUE),
    sd_difference = sd(score[group == "expert"] - score[group == "llms"], na.rm = TRUE)
  )

# 显示配对t检验结果
print("配对样本t检验结果:")
print(paired_t_test_results)

# 绘制箱线图比较两组在7个指标上的得分
ggplot(long_data, aes(x = measure, y = score, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("expert" = "#E69F00", "llms" = "#56B4E9")) +
  labs(
    title = "专家评分与LLMS评分在7个指标上的比较",
    x = "评估指标",
    y = "评分",
    fill = "组别"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

# 绘制均值比较图
mean_plot_data <- long_data %>%
  group_by(measure, group) %>%
  summarise(mean_score = mean(score, na.rm = TRUE),
            se = sd(score, na.rm = TRUE) / sqrt(n()))

ggplot(mean_plot_data, aes(x = measure, y = mean_score, color = group, group = group)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_score - se, ymax = mean_score + se), 
                width = 0.2, size = 0.8) +
  scale_color_manual(values = c("expert" = "#E69F00", "llms" = "#56B4E9")) +
  labs(
    title = "",
    x = "Evaluation Metrics",
    y = "Mean Score",
    color = "Group"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

# 绘制配对差异图（新增）
# 创建配对差异数据
expert_long$score<- as.numeric(expert_long$score)
llms_long$score<- as.numeric(llms_long$score)
difference_data <- expert_long %>%
  rename(expert_score = score) %>%
  left_join(llms_long %>% 
              rename(llms_score = score) %>% 
              select(id, measure, llms_score),
            by = c("id", "measure")) %>%
  mutate(difference = expert_score - llms_score)

ggplot(difference_data, aes(x = measure, y = difference)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "专家评分与LLMS评分的配对差异",
    x = "评估指标",
    y = "差异分数 (专家 - LLMS)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# 可选：保存配对t检验结果到CSV文件
write.csv(paired_t_test_results, "paired_t_test_results_expert_vs_llms.csv", row.names = FALSE)

# 显示简要统计摘要
cat("\n简要统计摘要:\n")
summary_stats <- long_data %>%
  group_by(measure, group) %>%
  summarise(
    n = n(),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE),
    min = min(score, na.rm = TRUE),
    max = max(score, na.rm = TRUE)
  ) %>%
  arrange(measure, group)

print(summary_stats)

# 显示配对差异统计
cat("\n配对差异统计:\n")
difference_stats <- difference_data %>%
  group_by(measure) %>%
  summarise(
    n_pairs = n(),
    mean_difference = mean(difference, na.rm = TRUE),
    sd_difference = sd(difference, na.rm = TRUE),
    min_difference = min(difference, na.rm = TRUE),
    max_difference = max(difference, na.rm = TRUE)
  )

print(difference_stats)