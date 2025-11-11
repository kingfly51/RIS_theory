library(readxl)
data <- read_excel("D:/Rdaima/RIS_theory/331954276_按文本_大学生休息羞耻现状调查_469_469.xlsx")
#data <- read_excel("D:/Rdaima/RIS_theory/331954276_按文本_大学生休息羞耻现状调查_746_746.xlsx")
data$reason_RIS<-data$reason

#计算ris-8总分
colnames(data)[7:14] <- paste0("ris", 1:8)

library(dplyr)
data <- data %>%
  mutate(across(ris1:ris8, ~ case_when(
    . == "非常不同意" ~ 1,
    . == "不同意" ~ 2,
    . == "一般" ~ 3,
    . == "同意" ~ 4,
    . == "非常同意" ~ 5,
    TRUE ~ as.numeric(.)
  ))) %>%
  mutate(ris_total = rowSums(across(ris1:ris8), na.rm = TRUE))

library(psych)
#NEGATIVE FEELING
NF_items <- data[, c("ris6", "ris8")] 
alpha_ris1 <- alpha(NF_items)
print(alpha_ris1)#0.82
#sOCIAL COMPARE
SC_items <- data[, c("ris1", "ris2")] 
alpha_ris2 <- alpha(SC_items)
print(alpha_ris2)#0.87
#ot
OT_items <- data[, c("ris5", "ris7")] 
alpha_ris3 <- alpha(OT_items)
print(alpha_ris3)#0.79
#CB
CB_items <- data[, c("ris3", "ris4")] 
alpha_ris4 <- alpha(CB_items)
print(alpha_ris4)#0.69
#OVERALL
ALL_items <- data[, c("ris1", "ris2","ris3", "ris4","ris5", "ris6","ris7", "ris8")] 
alpha_ris5 <- alpha(ALL_items)
print(alpha_ris5)#0.69
