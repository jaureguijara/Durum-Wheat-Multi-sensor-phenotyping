# --- Load required libraries ---
library(lme4)
library(emmeans)
library(dplyr)
library(ggplot2)
library(readr)
library(Matrix)

setwd("C:/Users/jaure/OneDrive - Universitat de Barcelona (1)/UB/Doctorat/Holistic Wheat/analysis/datasets")

df <- read.csv('holistic_wheat_merged_multi.csv')
df <- df[(df['Date'] == '2024-05-14') | (df['Date'] == '2024-05-29'),]

df_y <-read.csv('holistic_wheat_yield_merged.csv')

df$PN <- as.integer(df$PN)
df_y$PN <- as.integer(df_y$PN)

colnames(df_y)[8] <- "Yield_kg_ha"


df <- merge(
  df,
  df_y[, c("Loc", "PN", "Treatment", "Code", "Yield_kg_ha")],
  by = c("Loc", "PN", "Treatment", "Code"),
  all.x = TRUE
)

df$Loc_Treatment <- paste(df$Loc, df$Treatment, sep = "_")

df <- df[!is.na(df[["Yield_kg_ha"]]), ]


# --- Load your data (replace path if needed) ---
# df <- read_csv("your_data.csv")

# --- Create Environment identifier for G×E interaction ---
df <- df %>%
  mutate(
    Environment = as.factor(Loc_Treatment),
    Code = as.factor(Code),
    Location = as.factor(Loc),
    Treatment = as.factor(Treatment),
    Block = as.factor(B)
  )

# --- Fit Linear Mixed Model ---

model <- lmer(
  Yield_kg_ha ~ Code + 
    (1 | Location) + 
    (1 | Treatment) + 
    (1 | Location:Treatment) +
    (1 | Location:Treatment:Block) + 
    (1 | Code:Environment),
  data = df
)

# --- Model summary ---
cat(rep("=", 80), "\n")
cat("MODEL SUMMARY\n")
cat(rep("=", 80), "\n")
print(summary(model))

# --- Random effects variance components ---
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("RANDOM EFFECTS VARIANCE COMPONENTS\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

print(VarCorr(model), comp = "Variance")

# --- Estimated Marginal Means (EMMs) per genotype ---
emm_df <- as.data.frame(emmeans(model, specs = "Code"))
head(emm_df)

# --- Classify genotypes into yield groups (quartiles) ---
upper_q <- quantile(emm_df$emmean, 0.75)
lower_q <- quantile(emm_df$emmean, 0.25)

emm_df <- emm_df %>%
  mutate(
    Yield_group = case_when(
      emmean >= upper_q ~ "High yield",
      emmean <= lower_q ~ "Low yield",
      TRUE ~ "Intermediate yield"
    )
  )

# --- Save EMM results ---
write_csv(emm_df, "../results/yield_groups_based_on_emms.csv")
cat("\nEMMs saved to 'yield_groups_based_on_emms.csv'\n")

# --- Visualization of Yield Groups ---
ggplot(emm_df, aes(x = Yield_group, y = emmean, fill = Yield_group)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_brewer(palette = "Pastel2") +
  labs(
    title = "Yield Groups Based on Estimated Marginal Means",
    x = "Yield Group",
    y = "Estimated Marginal Mean (Yield, kg/ha)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("../results/figures/yield_groups_boxplot.png", width = 8, height = 5, dpi = 300)

# --- Summary statistics per yield group ---
summary_stats <- emm_df %>%
  group_by(Yield_group) %>%
  summarise(
    mean = mean(emmean),
    sd = sd(emmean),
    count = n(),
    se = sd / sqrt(count)
  )

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("YIELD GROUP STATISTICS\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

print(summary_stats)

for (i in 1:nrow(summary_stats)) {
  cat(
    sprintf(
      "%-20s Mean = %.2f kg/ha, SD = %.2f, SE = %.2f, n = %d\n",
      summary_stats$Yield_group[i],
      summary_stats$mean[i],
      summary_stats$sd[i],
      summary_stats$se[i],
      summary_stats$count[i]
    )
  )
}

cat("\n", paste(rep("=", 80), collapse = ""), "\n")

