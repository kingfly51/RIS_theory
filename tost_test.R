# 首次使用时安装：
# install.packages(c("readxl", "writexl"))

library(readxl)
library(writexl)

# 1. 读取数据
input_file <- "D:/Rdaima/RIS_theory/data/sample2/combined_data62.xlsx"
dat <- read_excel(input_file)

dimensions <- c(
  "Logical_coherence", "Explain_depth", "comprehensiveness",
  "Persuasion", "real-life experience", "clarity", "heuristic"
)

# 2. 转换评分为数值，并检查异常
for (name in c(paste0("expert_", dimensions),
               paste0("llms_", dimensions))) {
  if (!name %in% names(dat)) stop("缺少列：", name)
  
  raw <- trimws(as.character(dat[[name]]))
  raw[!is.na(raw) & raw == ""] <- NA_character_
  x <- suppressWarnings(as.numeric(raw))
  
  if (any(!is.na(raw) & (is.na(x) | !is.finite(x))) ||
      any(x < 1 | x > 9, na.rm = TRUE)) {
    stop("评分无效或超出1—9分：", name)
  }
  dat[[name]] <- x
}

# 3. 分别检验 ±0.5分和 ±1分
results <- do.call(rbind, lapply(c(0.5, 1), function(margin) {
  
  out <- do.call(rbind, lapply(dimensions, function(dimension) {
    
    expert <- dat[[paste0("expert_", dimension)]]
    llm <- dat[[paste0("llms_", dimension)]]
    
    keep <- complete.cases(expert, llm)
    expert <- expert[keep]
    llm <- llm[keep]
    difference <- llm - expert
    
    n <- length(difference)
    if (n < 2) stop(dimension, "：有效配对不足")
    if (sd(difference) == 0) stop(dimension, "：差值无变异")
    
    m <- mean(difference)
    se <- sd(difference) / sqrt(n)
    
    # 两个单侧检验
    p_lower <- pt((m + margin) / se, df = n - 1,
                  lower.tail = FALSE)
    p_upper <- pt((m - margin) / se, df = n - 1)
    
    data.frame(
      Dimension = dimension,
      Margin = margin,
      N = n,
      Expert_mean = mean(expert),
      LLM_mean = mean(llm),
      Difference = m,                  # LLM - Expert
      CI90_lower = m - qt(0.95, n - 1) * se,
      CI90_upper = m + qt(0.95, n - 1) * se,
      p_TOST = max(p_lower, p_upper)
    )
  }))
  
  # 每个界值内，对七个维度进行Holm校正
  out$p_Holm <- p.adjust(out$p_TOST, method = "holm")
  out$Equivalent <- out$p_TOST < 0.05
  out$Equivalent_Holm <- out$p_Holm < 0.05
  out
}))

# 4. 显示并保存
print(results, row.names = FALSE, digits = 4)

write_xlsx(
  results,
  file.path(dirname(input_file), "TOST_0.5_and_1.xlsx")
)