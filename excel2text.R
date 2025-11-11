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


data_high <- data[data$ris_total>28,]
library(writexl)
write_xlsx(data_high,"D:/Rdaima/RIS_theory/data_high_746.xlsx")



text_column <- "reason_RIS"
output_dir <- "D:/Rdaima/RIS_theory/reason_RIS_high_623"

for(i in 1:nrow(data_high)) {
  filename <- sprintf("sub%03d.txt", i)
  filepath <- file.path(output_dir, filename)
  text_content <- as.character(data_high[i, text_column])
  writeLines(text_content, filepath)
}
