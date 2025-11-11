# 加载必要的包
library(readxl)
library(lavaan)
library(dplyr)
library(tidyr)

# 读取数据
theory_data <- read_excel("D:/Rdaima/RIS_theory/验证理论数据/337027702_按文本_大学生休息羞耻影响因素调查_392_392.xlsx")
colnames(theory_data)[8:39] <- c("RIS1",
                                 "RIS2",
                                 "RIS3",
                                 "RIS4",
                                 "RIS5",
                                 "RIS6",
                                 "RIS7",
                                 "RIS8",
                                 "IPV1",
                                 "IPV2",
                                 "IPV3",
                                 "IPV4",
                                 "PCSC1",
                                 "PCSC2",
                                 "PCSC3",
                                 "PCSC4",
                                 "IDCI1",
                                 "IDCI2",
                                 "IDCI3",
                                 "IDCI4",
                                 "TTMP1",
                                 "TTMP2",
                                 "TTMP3",
                                 "TTMP4",
                                 "DSPER1",
                                 "DSPER2",
                                 "DSPER3",
                                 "DSPER4",
                                 "DEEV1",
                                 "DEEV2",
                                 "DEEV3",
                                 "DEEV4")

# 数据清洗和转换
theory_data_clean <- theory_data %>%
  # 替换第8-39列的评分值
  mutate(across(8:39, ~ case_when(
    . == "非常不同意" ~ 1,
    . == "比较不同意" ~ 2,
    . == "既不同于也不反对" ~ 3,
    . == "比较同意" ~ 4,
    . == "非常同意" ~ 5,
    TRUE ~ as.numeric(.)
  ))) %>%
  # 处理所用时间列：提取秒数并转换为数值
  mutate(
    所用时间 = as.numeric(gsub("秒", "", 所用时间))
  ) %>%
  # 筛选时间大于80秒的被试
  filter(所用时间 > 60)

# 计算各量表总分
theory_data_scores <- theory_data_clean %>%
  mutate(
    # IPV总分 (IPV1+IPV2+IPV3+IPV4)
    IPV_total = rowSums(select(., matches("^IPV[1-4]$")), na.rm = TRUE),
    
    # PCSC总分 (PCSC1+PCSC2+PCSC3+PCSC4)
    PCSC_total = rowSums(select(., matches("^PCSC[1-4]$")), na.rm = TRUE),
    
    # IDCI总分 (IDCI1+IDCI2+IDCI3+IDCI4)
    IDCI_total = rowSums(select(., matches("^IDCI[1-4]$")), na.rm = TRUE),
    
    # TTMP总分 (TTMP1+TTMP2+TTMP3+TTMP4)
    TTMP_total = rowSums(select(., matches("^TTMP[1-4]$")), na.rm = TRUE),
    
    # DSPER总分 (DSPER1+DSPER2+DSPER3+DSPER4)
    DSPER_total = rowSums(select(., matches("^DSPER[1-4]$")), na.rm = TRUE),
    
    # DEEV总分 (DEEV1+DEEV2+DEEV3+DEEV4)
    DEEV_total = rowSums(select(., matches("^DEEV[1-4]$")), na.rm = TRUE),
    
    # RIS总分 (RIS1+RIS2+...+RIS8)
    RIS_total = rowSums(select(., matches("^RIS[1-8]$")), na.rm = TRUE)
  )

# 中心化处理
theory_data_centered <- theory_data_scores %>%
  mutate(
    # 对主要预测变量进行中心化
    IPV_total_c = scale(IPV_total, center = TRUE, scale = FALSE),
    DSPER_total_c = scale(DSPER_total, center = TRUE, scale = FALSE),
    DEEV_total_c = scale(DEEV_total, center = TRUE, scale = FALSE),
    TTMP_total_c = scale(TTMP_total, center = TRUE, scale = FALSE),
    PCSC_total_c = scale(PCSC_total, center = TRUE, scale = FALSE),
    IDCI_total_c = scale(IDCI_total, center = TRUE, scale = FALSE),
    
    # 创建中心化后的交互项
    IPV_DSPER_int = IPV_total_c * DSPER_total_c,
    DEEV_DSPER_int = DEEV_total_c * DSPER_total_c,
    TTMP_DSPER_int = TTMP_total_c * DSPER_total_c,
    PCSC_DSPER_int = PCSC_total_c * DSPER_total_c
  )
writexl::write_xlsx(theory_data_centered,"theory_data_centered.xlsx")
# 检查数据
cat("数据维度:", dim(theory_data_centered), "\n")
cat("样本量:", nrow(theory_data_centered), "\n")

# 描述性统计
cat("\n各变量描述性统计（中心化前）:\n")
theory_data_scores %>%
  select(IPV_total, PCSC_total, IDCI_total, TTMP_total, 
         DSPER_total, DEEV_total, RIS_total) %>%
  summary()

cat("\n各变量描述性统计（中心化后）:\n")
theory_data_centered %>%
  select(IPV_total_c, PCSC_total_c, IDCI_total_c, TTMP_total_c, 
         DSPER_total_c, DEEV_total_c, RIS_total) %>%
  summary()

# 构建结构方程模型（使用中心化变量）
model_total <- '
  # 结构模型（使用中心化变量）
  # IDCI指向其他变量
  IPV_total_c ~ a1 * IDCI_total_c
  DEEV_total_c ~ a2 * IDCI_total_c
  TTMP_total_c ~ a3 * IDCI_total_c
  PCSC_total_c ~ a4 * IDCI_total_c
  
  # IPV指向其他变量
  DEEV_total_c ~ b1 * IPV_total_c
  TTMP_total_c ~ b2 * IPV_total_c
  PCSC_total_c ~ b3 * IPV_total_c
  RIS_total ~ b4 * IPV_total_c
  
  # DEEV, TTMP, PCSC指向RIS
  RIS_total ~ c1 * DEEV_total_c + c2 * TTMP_total_c + c3 * PCSC_total_c
  
  # DSPER的主效应和交互效应（使用预计算的交互项）
  RIS_total ~ d1 * DSPER_total_c + 
              d2 * IPV_DSPER_int + 
              d3 * DEEV_DSPER_int + 
              d4 * TTMP_DSPER_int + 
              d5 * PCSC_DSPER_int
  
  
  # 外生变量间的相关
  IDCI_total_c ~~ DSPER_total_c
  
  # 中介变量间的相关
  DEEV_total_c ~~ TTMP_total_c + PCSC_total_c
  TTMP_total_c ~~ PCSC_total_c

  
  # 交互项间的相关（同源交互项可能相关）
  IPV_DSPER_int ~~ DEEV_DSPER_int + TTMP_DSPER_int + PCSC_DSPER_int
  DEEV_DSPER_int ~~ TTMP_DSPER_int + PCSC_DSPER_int
  TTMP_DSPER_int ~~ PCSC_DSPER_int
 
   # ========== 修正相关 ==========
  DEEV_total_c ~~  DSPER_total_c
  IPV_total_c ~~  DSPER_total_c
  TTMP_total_c ~~  DSPER_total_c
  TTMP_total_c ~~ TTMP_DSPER_int
  IPV_total_c ~~ TTMP_DSPER_int
'

# 拟合模型
fit_total <- sem(model_total, data = theory_data_centered)


# 查看模型结果
cat("=== 模型拟合结果（中心化变量） ===\n")
summary(fit_total, standardized = TRUE, fit.measures = TRUE, rsq = TRUE)

# 详细的拟合指标
cat("\n=== 模型拟合指标 ===\n")
fit_measures <- fitMeasures(fit_total)
print(fit_measures[c("chisq", "df", "pvalue", "cfi","nfi","gfi","rfi", "tli", "rmsea", "srmr")])

# 参数估计结果
cat("\n=== 参数估计结果 ===\n")
parameter_estimates <- parameterEstimates(fit_total, standardized = TRUE)
print(parameter_estimates)

# 加载必要的包
library(ggplot2)
library(interactions)
#library(sjPlot)

# 提取PCSC*DSPER调节效应结果
pcsc_moderation <- parameterEstimates(fit_total, standardized = TRUE) %>%
  filter(op == "~" & lhs == "RIS_total" & rhs == "PCSC_DSPER_int")

cat("=== PCSC × DSPER 调节效应结果 ===\n")
print(pcsc_moderation)

# 简单效应检验 - 使用回归方法
cat("\n=== PCSC × DSPER 简单斜率分析 ===\n")

# 构建包含所有控制变量的回归模型
model_pcsc <- lm(RIS_total ~ PCSC_total_c * DSPER_total_c + 
                   IPV_total_c + DEEV_total_c + TTMP_total_c + 
                   IDCI_total_c, 
                 data = theory_data_centered)

# 模型摘要
summary(model_pcsc)

# 简单斜率分析
cat("\n--- 简单斜率分析结果 ---\n")
pcsc_simple <- sim_slopes(model_pcsc, 
                          pred = PCSC_total_c, 
                          modx = DSPER_total_c,
                          jnplot = TRUE)
print(pcsc_simple)

# Johnson-Neyman分析
cat("\n--- Johnson-Neyman分析 ---\n")
pcsc_jn <- johnson_neyman(model_pcsc, 
                          pred = PCSC_total_c, 
                          modx = DSPER_total_c)
print(pcsc_jn)

# 绘制调节效应图
cat("\n=== 绘制调节效应图 ===\n")

# 方法1: 使用interactions包
p1 <- interact_plot(model_pcsc, 
                    pred = PCSC_total_c, 
                    modx = DSPER_total_c,
                    modx.values = c(-1, 0, 1),  # 使用标准差单位
                    x.label = "PCSC (Center)",
                    y.label = "RIS total score",
                    modx.labels = c("Low DSPER (-1 SD)", "Mean DSPER", "High DSPER (+1 SD)"),
                    legend.main = "DSPER Level",
                    colors = c("#E41A1C", "#377EB8", "#4DAF4A")) +
  ggtitle("") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p1)



# 重新拟合模型，使用Bootstrap估计
fit_bootstrap <- sem(model_total, 
                     data = theory_data_centered,
                     se = "bootstrap", 
                     bootstrap = 5000,
                     parallel = "snow",  # 并行计算加速
                     ncpus = parallel::detectCores() - 1)  # 使用多核
summary(fit_bootstrap)
fit_measures <- fitMeasures(fit_bootstrap)
print(fit_measures[c("chisq", "df", "pvalue", "cfi","nfi","gfi","rfi", "tli", "rmsea", "srmr")])

# 定义间接效应（与之前相同）
indirect_effects_boot <- '
  # IDCI通过IPV对DEEV的间接效应
  indirect_IDCI_IPV_DEEV := a1 * b1
  
  # IDCI通过IPV对TTMP的间接效应
  indirect_IDCI_IPV_TTMP := a1 * b2
  
  # IDCI通过IPV对PCSC的间接效应
  indirect_IDCI_IPV_PCSC := a1 * b3
  
  # IDCI通过IPV对RIS的间接效应
  indirect_IDCI_IPV_RIS := a1 * b4
  
  # IDCI通过DEEV对RIS的间接效应
  indirect_IDCI_DEEV_RIS := a2 * c1
  
  # IDCI通过TTMP对RIS的间接效应
  indirect_IDCI_TTMP_RIS := a3 * c2
  
  # IDCI通过PCSC对RIS的间接效应
  indirect_IDCI_PCSC_RIS := a4 * c3
  
  # IDCI通过各中介变量对RIS的总间接效应
  total_indirect_IDCI_RIS := a1*b4 + a2*c1 + a3*c2 + a4*c3
  
  # IDCI对RIS的总效应
  total_IDCI_RIS := a1*b4 + a2*c1 + a3*c2 + a4*c3
'

# 计算Bootstrap间接效应
fit_indirect_boot <- sem(paste(model_total, indirect_effects_boot), 
                         data = theory_data_centered,
                         se = "bootstrap",
                         bootstrap = 5000,
                         parallel = "snow",
                         ncpus = parallel::detectCores() - 1)

cat("Bootstrap间接效应结果 (5000次抽样):\n")
indirect_results_boot <- parameterEstimates(fit_indirect_boot, 
                                            boot.ci.type = "bca.simple")  # 偏差校正的置信区间

# 显示间接效应结果
defined_params_boot <- indirect_results_boot[!is.na(indirect_results_boot$label), ]
print(defined_params_boot[, c("label", "est", "se", "pvalue", "ci.lower", "ci.upper")])


# 模型诊断
cat("\n=== 模型诊断 ===\n")
cat("标准化残差的绝对值大于2的个案:\n")
residuals <- resid(fit_total)$cov
large_resid <- which(abs(residuals) > 2, arr.ind = TRUE)
if(length(large_resid) > 0) {
  print(large_resid)
} else {
  cat("没有发现大的标准化残差\n")
}

# 检查多重共线性
cat("\n=== 多重共线性检查 ===\n")
if(require(car)) {
  # 创建回归模型检查VIF
  lm_check <- lm(RIS_total ~ IPV_total_c + DEEV_total_c + TTMP_total_c + 
                   PCSC_total_c + DSPER_total_c + IPV_DSPER_int + 
                   DEEV_DSPER_int + TTMP_DSPER_int + PCSC_DSPER_int, 
                 data = theory_data_centered)
  vif_values <- vif(lm_check)
  cat("方差膨胀因子(VIF):\n")
  print(vif_values)
  cat("注：VIF > 10 表示严重多重共线性\n")
}

# 可视化路径系数
cat("\n生成路径图...\n")
library(semPlot)
semPaths(fit_total, 
         whatLabels = "std", 
         layout = "tree", 
         style = "lisrel", 
         edge.label.cex = 0.8, 
         sizeMan = 8,
         nCharNodes = 0,
         rotation = 2)

# 保存结果
output <- list(
  model_summary = summary(fit_total, standardized = TRUE),
  fit_measures = fitMeasures(fit_total),
  parameter_estimates = parameterEstimates(fit_total, standardized = TRUE),
  data = theory_data_centered,
  centered_stats = psych::describe(theory_data_centered %>% 
                                     select(ends_with("_c"), ends_with("_int"), RIS_total))
)





# 定义各量表的项目
scales <- list(
  IPV = c("IPV1", "IPV2", "IPV3", "IPV4"),
  PCSC = c("PCSC1", "PCSC2", "PCSC3", "PCSC4"),
  IDCI = c("IDCI1", "IDCI2", "IDCI3", "IDCI4"),
  TTMP = c("TTMP1", "TTMP2", "TTMP3", "TTMP4"),
  DSPER = c("DSPER1", "DSPER2", "DSPER3", "DSPER4"),
  DEEV = c("DEEV1", "DEEV2", "DEEV3", "DEEV4"),
  RIS = c("RIS1", "RIS2", "RIS3", "RIS4", "RIS5", "RIS6", "RIS7", "RIS8")
)

# ==================== 内部一致性分析 ====================
cat("=== 各量表内部一致性系数（Cronbach's Alpha） ===\n\n")

alpha_results <- list()

for(scale_name in names(scales)) {
  cat("量表:", scale_name, "\n")
  items <- scales[[scale_name]]
  
  # 检查项目是否存在
  if(all(items %in% colnames(theory_data_clean))) {
    scale_data <- theory_data_clean[, items]
    
    # 计算Cronbach's Alpha
    alpha_result <- psych::alpha(scale_data, check.keys = TRUE)
    alpha_results[[scale_name]] <- alpha_result
    
    cat("项目数:", length(items), "\n")
    cat("Cronbach's Alpha:", round(alpha_result$total$raw_alpha, 3), "\n")
    
    # 判断标准
    alpha_value <- alpha_result$total$raw_alpha
    if(alpha_value >= 0.8) {
      cat("信度: 优秀\n")
    } else if(alpha_value >= 0.7) {
      cat("信度: 可接受\n")
    } else if(alpha_value >= 0.6) {
      cat("信度: 一般\n")
    } else {
      cat("信度: 不足\n")
    }
    
    # 显示项目-总分相关
    cat("项目-总分相关:\n")
    item_stats <- alpha_result$item.stats
    print(round(item_stats[, c("r.cor", "r.drop")], 3))
    cat("---\n\n")
  } else {
    cat("错误: 部分项目不存在\n\n")
  }
}

# ==================== 验证性因素分析 ====================
cat("=== 各量表验证性因素分析 ===\n\n")

cfa_results <- list()

# 对每个量表单独进行CFA
for(scale_name in names(scales)) {
  cat("========== ", scale_name, "量表CFA分析 ==========\n")
  items <- scales[[scale_name]]
  
  if(all(items %in% colnames(theory_data_clean))) {
    # 构建CFA模型
    cfa_model <- paste0(scale_name, " =~ ", paste(items, collapse = " + "))
    
    cat("CFA模型:\n", cfa_model, "\n\n")
    
    # 拟合CFA模型
    cfa_fit <- cfa(cfa_model, 
                   data = theory_data_clean,
                   estimator = "ML",
                   missing = "ml")
    
    cfa_results[[scale_name]] <- cfa_fit
    
    # 显示拟合指标
    cat("拟合指标:\n")
    fit_measures <- fitMeasures(cfa_fit)
    key_measures <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "rmsea.ci.lower", 
                      "rmsea.ci.upper", "srmr")
    fit_summary <- fit_measures[key_measures]
    print(round(fit_summary, 3))
    
    # 拟合判断
    cat("\n拟合判断:\n")
    if(fit_summary["cfi"] >= 0.95 & fit_summary["tli"] >= 0.95 & fit_summary["rmsea"] <= 0.06) {
      cat("拟合: 优秀\n")
    } else if(fit_summary["cfi"] >= 0.90 & fit_summary["tli"] >= 0.90 & fit_summary["rmsea"] <= 0.08) {
      cat("拟合: 可接受\n")
    } else {
      cat("拟合: 需要改进\n")
    }
    
    # 显示因子载荷
    cat("\n标准化因子载荷:\n")
    loadings <- parameterEstimates(cfa_fit, standardized = TRUE)
    loadings <- loadings[loadings$op == "=~", ]
    print(loadings[, c("rhs", "est", "std.all", "se", "pvalue")])
    
    # 载荷判断
    cat("\n因子载荷判断:\n")
    low_loadings <- sum(loadings$std.all < 0.5)
    cat("载荷 < 0.5 的项目数:", low_loadings, "\n")
    cat("平均载荷:", round(mean(loadings$std.all), 3), "\n")
    
    # 组合信度(CR)和平均变异抽取量(AVE)
    cat("\n信度和效度指标:\n")
    loadings_vec <- loadings$std.all
    cr <- (sum(loadings_vec))^2 / ((sum(loadings_vec))^2 + sum(1 - loadings_vec^2))
    ave <- sum(loadings_vec^2) / length(loadings_vec)
    
    cat("组合信度(CR):", round(cr, 3), "\n")
    cat("平均变异抽取量(AVE):", round(ave, 3), "\n")
    
    if(cr >= 0.7) {
      cat("组合信度: 良好\n")
    } else if(cr >= 0.6) {
      cat("组合信度: 可接受\n")
    } else {
      cat("组合信度: 不足\n")
    }
    
    if(ave >= 0.5) {
      cat("收敛效度: 良好\n")
    } else {
      cat("收敛效度: 不足\n")
    }
    
    cat("\n", strrep("-", 50), "\n\n")
    
  } else {
    cat("错误: 部分项目不存在\n\n")
  }
}

# ==================== 验证性因素分析结果整理到表格 ====================
cat("=== 各量表验证性因素分析结果汇总表 ===\n\n")

# 创建CFA结果汇总表
cfa_summary_table <- data.frame(
  量表名称 = character(),
  项目数 = integer(),
  χ= numeric(),
  自由度 = numeric(),
  p值 = numeric(),
  CFI = numeric(),
  TLI = numeric(),
  RMSEA = numeric(),
  RMSEA_90CI = character(),
  SRMR = numeric(),
  平均因子载荷 = numeric(),
  组合信度_CR = numeric(),
  平均变异抽取量_AVE = numeric(),
  拟合优度 = character(),
  字符串形式 = character(),
  stringsAsFactors = FALSE
)

# 创建因子载荷明细表
factor_loadings_table <- data.frame(
  量表名称 = character(),
  项目 = character(),
  非标准化载荷 = numeric(),
  标准化载荷 = numeric(),
  标准误 = numeric(),
  p值 = numeric(),
  显著性 = character(),
  stringsAsFactors = FALSE
)

# 遍历每个量表，收集CFA结果
for(scale_name in names(scales)) {
  items <- scales[[scale_name]]
  
  if(all(items %in% colnames(theory_data_clean)) && scale_name %in% names(cfa_results)) {
    cfa_fit <- cfa_results[[scale_name]]
    
    # 提取拟合指标
    fit_measures <- fitMeasures(cfa_fit)
    chisq <- fit_measures["chisq"]
    df <- fit_measures["df"]
    pvalue <- fit_measures["pvalue"]
    cfi <- fit_measures["cfi"]
    tli <- fit_measures["tli"]
    rmsea <- fit_measures["rmsea"]
    rmsea_lower <- fit_measures["rmsea.ci.lower"]
    rmsea_upper <- fit_measures["rmsea.ci.upper"]
    srmr <- fit_measures["srmr"]
    
    # 提取因子载荷
    loadings <- parameterEstimates(cfa_fit, standardized = TRUE)
    loadings <- loadings[loadings$op == "=~", ]
    avg_loading <- mean(loadings$std.all)
    
    # 计算CR和AVE
    loadings_vec <- loadings$std.all
    cr <- (sum(loadings_vec))^2 / ((sum(loadings_vec))^2 + sum(1 - loadings_vec^2))
    ave <- sum(loadings_vec^2) / length(loadings_vec)
    
    # 判断拟合优度
    if(cfi >= 0.95 & tli >= 0.95 & rmsea <= 0.06) {
      fit_quality <- "优秀"
    } else if(cfi >= 0.90 & tli >= 0.90 & rmsea <= 0.08) {
      fit_quality <- "可接受"
    } else {
      fit_quality <- "需要改进"
    }
    
    # 添加到汇总表
    cfa_summary_table <- rbind(cfa_summary_table, data.frame(
      量表名称 = scale_name,
      项目数 = length(items),
      χ = round(chisq, 3),
      自由度 = round(df, 0),
      p值 = round(pvalue, 3),
      CFI = round(cfi, 3),
      TLI = round(tli, 3),
      RMSEA = round(rmsea, 3),
      RMSEA_90CI = paste0("[", round(rmsea_lower, 3), ", ", round(rmsea_upper, 3), "]"),
      SRMR = round(srmr, 3),
      平均因子载荷 = round(avg_loading, 3),
      组合信度_CR = round(cr, 3),
      平均变异抽取量_AVE = round(ave, 3),
      拟合优度 = fit_quality,
      字符串形式 = paste(items, collapse = ", "),
      stringsAsFactors = FALSE
    ))
    
    # 添加到因子载荷明细表
    for(i in 1:nrow(loadings)) {
      loading_row <- loadings[i, ]
      significance <- ifelse(loading_row$pvalue < 0.001, "***",
                             ifelse(loading_row$pvalue < 0.01, "**",
                                    ifelse(loading_row$pvalue < 0.05, "*", "ns")))
      
      factor_loadings_table <- rbind(factor_loadings_table, data.frame(
        量表名称 = scale_name,
        项目 = loading_row$rhs,
        非标准化载荷 = round(loading_row$est, 3),
        标准化载荷 = round(loading_row$std.all, 3),
        标准误 = round(loading_row$se, 3),
        p值 = round(loading_row$pvalue, 3),
        显著性 = significance,
        stringsAsFactors = FALSE
      ))
    }
  }
}

# 显示CFA汇总表
cat("表1: 验证性因素分析拟合指标汇总\n")
print(cfa_summary_table)

# 显示因子载荷明细表
cat("\n表2: 各量表因子载荷明细\n")
print(factor_loadings_table)

# 创建信效度指标判断表
reliability_validity_table <- data.frame(
  量表名称 = cfa_summary_table$量表名称,
  项目数 = cfa_summary_table$项目数,
  Cronbach_Alpha = NA,  # 需要从alpha_results中获取
  组合信度_CR = cfa_summary_table$组合信度_CR,
  平均变异抽取量_AVE = cfa_summary_table$平均变异抽取量_AVE,
  组合信度判断 = ifelse(cfa_summary_table$组合信度_CR >= 0.7, "良好",
                  ifelse(cfa_summary_table$组合信度_CR >= 0.6, "可接受", "不足")),
  收敛效度判断 = ifelse(cfa_summary_table$平均变异抽取量_AVE >= 0.5, "良好", "不足"),
  平均因子载荷 = cfa_summary_table$平均因子载荷,
  低载荷项目数 = NA,  # 需要计算
  stringsAsFactors = FALSE
)

# 填充Cronbach's Alpha和低载荷项目数
for(i in 1:nrow(reliability_validity_table)) {
  scale_name <- reliability_validity_table$量表名称[i]
  
  # 从alpha_results获取Cronbach's Alpha
  if(scale_name %in% names(alpha_results)) {
    reliability_validity_table$Cronbach_Alpha[i] <- round(alpha_results[[scale_name]]$total$raw_alpha, 3)
  }
  
  # 计算低载荷项目数
  scale_loadings <- factor_loadings_table[factor_loadings_table$量表名称 == scale_name, "标准化载荷"]
  reliability_validity_table$低载荷项目数[i] <- sum(scale_loadings < 0.5)
}

cat("\n表3: 信度和效度指标汇总\n")
print(reliability_validity_table)

# 创建多量表区分效度表（因子间相关）
cat("\n表4: 多量表因子间相关矩阵（区分效度）\n")
if(exists("multiscale_fit")) {
  factor_cor <- lavInspect(multiscale_fit, what = "cor.lv")
  factor_cor_table <- as.data.frame(round(factor_cor, 3))
  factor_cor_table$因子 <- rownames(factor_cor_table)
  factor_cor_table <- factor_cor_table[, c("因子", colnames(factor_cor))]
  print(factor_cor_table)
}

# 创建格式化输出用于论文
cat("\n=== 用于论文的格式化输出 ===\n\n")

# CFA拟合指标表（三线表格式）
cat("表1 验证性因素分析拟合指标\n")
cat("量表\tχ²\tdf\tCFI\tTLI\tRMSEA\tSRMR\tCR\tAVE\n")
for(i in 1:nrow(cfa_summary_table)) {
  row <- cfa_summary_table[i, ]
  cat(row$量表名称, "\t",
      row$χ², "\t", row$自由度, "\t",
      row$CFI, "\t", row$TLI, "\t",
      row$RMSEA, "\t", row$SRMR, "\t",
      row$组合信度_CR, "\t", row$平均变异抽取量_AVE, "\n")
}


