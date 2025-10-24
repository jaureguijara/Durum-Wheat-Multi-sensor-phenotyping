# ===============================================================
#  HERITABILITY ESTIMATION (H²) FOR MULTIPLE DATASETS
#  Using anthesis (df_an), grain filling (df_gf), and per-location differences (df_diff)
# ===============================================================

# --- Libraries ---
library(lme4)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(stringr)
library(ggplot2)
library(patchwork)

# ===============================================================
# 1. DATA PREPARATION
# ===============================================================

setwd("C:/Users/jaure/OneDrive - Universitat de Barcelona (1)/UB/Doctorat/Holistic Wheat/analysis")

# --- Load datasets ---
df <- read_csv("datasets/holistic_wheat_merged_multi.csv") %>% arrange(Date)
df_y <- read_csv("datasets/holistic_wheat_yield_merged.csv")
ggd <- read_csv("datasets/meteo/GDD_dates_all_locations.csv")
ETo <- read_csv("datasets/meteo/ETo_prec_all_locations.csv")

# --- Merge yield and meteorological data ---
df <- df %>%
  left_join(df_y %>% select(Loc, PN, Treatment, Code, `Yield (kg/ha)`),
            by = c("Loc", "PN", "Treatment", "Code")) %>%
  left_join(ggd %>% select(Date, GDD), by = "Date") %>%
  left_join(ETo %>% select(Date, Treatment, `ETo (mm)`, water_input_cumulative),
            by = c("Date", "Treatment"))

# --- Clean column names ---
names(df) <- gsub("\\*", "", names(df))
names(df) <- gsub("\\/", "_", names(df))

# --- Normalize temperature (Tveg_mean_UAV) within each Date × Loc ---
df <- df %>%
  group_by(Date, Loc) %>%
  mutate(Tveg_mean_UAV = (Tveg_mean_UAV - mean(Tveg_mean_UAV, na.rm = TRUE)) /
           sd(Tveg_mean_UAV, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(!is.na(`Yield (kg_ha)`))

# ===============================================================
# 2. DEFINE DATASETS
# ===============================================================

# --- Location-specific datasets for difference calculation ---
df_a  <- df %>% filter(Date %in% c("2024-04-22", "2024-05-14"))   # Aranjuez
df_v  <- df %>% filter(Date %in% c("2024-05-29", "2024-06-12"))   # Valladolid

# --- Development stage datasets ---
df_an <- df %>% filter(Date %in% c("2024-04-22", "2024-05-29"))   # Anthesis
df_gf <- df %>% filter(Date %in% c("2024-05-14", "2024-06-12"))   # Grain filling

# --- Identify columns ---
id_columns <- names(df)[1:11]
vi_all <- names(df)[12:48]
yield_column <- names(df)[49]
meteo_columns <- names(df)[50:length(names(df))]

select_vars <- c(id_columns, meteo_columns, vi_all, yield_column)
df_a  <- df_a[, select_vars]
df_v  <- df_v[, select_vars]
df_an <- df_an[, select_vars]
df_gf <- df_gf[, select_vars]

# ===============================================================
# 3. DIFFERENCE CALCULATION
# ===============================================================

calculate_vi_differences <- function(df, vi_columns) {
  df %>%
    arrange(PN, Treatment, Date) %>%
    group_by(PN, Treatment) %>%
    mutate(GDD_diff = GDD - lag(GDD)) %>%
    mutate(across(all_of(vi_columns), ~ (. - lag(.)) / GDD_diff, .names = "{.col}")) %>%
    ungroup() %>%
    drop_na(all_of(vi_columns))
}

# --- Differences within each location ---
df_a_diff <- calculate_vi_differences(df_a, vi_all)
df_v_diff <- calculate_vi_differences(df_v, vi_all)

# --- Combined difference dataset ---
df_diff <- bind_rows(df_a_diff, df_v_diff)


# ===============================================================
# 4. HERITABILITY CALCULATION FUNCTION
# ===============================================================
calculate_heritability <- function(df, dataset_label) {
  df <- df %>%
    mutate(
      Loc_Treatment = paste(Loc, Treatment, sep = "_"),
      Environment = as.factor(Loc_Treatment),
      Code = as.factor(Code),
      Location = as.factor(Loc),
      Treatment = as.factor(Treatment),
      Block = as.factor(B),
      # Create unique block identifiers
      Block_Location = as.factor(paste(Loc, B, sep = "_")),
      Block_Treatment = as.factor(paste(Treatment, B, sep = "_")),
      Block_Environment = as.factor(paste(Loc, Treatment, B, sep = "_"))
    )
  
  vi_cols <- vi_all
  results_all <- list()
  
  # Control settings for better convergence
  ctrl <- lmerControl(optimizer = "bobyqa",
                      optCtrl = list(maxfun = 100000),
                      calc.derivs = FALSE)
  
  # === OVERALL H² ===
  results_all$overall <- map_dfr(vi_cols, function(vi) {
    form <- as.formula(paste0(
      vi, " ~ 1 + (1|Code) + (1|Location) + (1|Treatment) + ",
      "(1|Location:Treatment) + (1|Block:Location:Treatment) + ",
      "+ (1|Code:Location:Treatment)"
    ))
    fit <- tryCatch(
      suppressWarnings(lmer(form, data = df, REML = TRUE, control = ctrl)), 
      error = function(e) NULL
    )
    if (is.null(fit)) return(tibble(VI = vi, Dataset = dataset_label, Condition = "Overall", H2 = NA))
    
    # Check for convergence issues
    if (!is.null(fit@optinfo$conv$lme4$code) && fit@optinfo$conv$lme4$code != 0) {
      return(tibble(VI = vi, Dataset = dataset_label, Condition = "Overall", H2 = NA))
    }
    
    vc <- as.data.frame(VarCorr(fit))
    resid_var <- attr(VarCorr(fit), "sc")^2
    get_v <- function(name) ifelse(name %in% vc$grp, vc$vcov[vc$grp == name], 0)
    vg <- get_v("Code")
    vgL <- get_v("Code:Location")
    vgE <- get_v("Code:Location:Treatment")
    
    nL <- n_distinct(df$Location)
    nE <- n_distinct(df$Environment)
    nrep <- mean(table(df$Code, df$Environment) > 0)
    
    denom <- vg + vgL/nL + vgE/nE + resid_var/(nE*nrep)
    h2 <- ifelse(denom > 0, vg/denom, NA)
    
    tibble(VI = vi, Dataset = dataset_label, Condition = "Overall", H2 = h2)
  })
  
  # === H² PER LOCATION ===
  locations <- unique(df$Location)
  results_all$by_location <- map_dfr(locations, function(loc) {
    data_subset <- df %>% filter(Location == loc)
    
    map_dfr(vi_cols, function(vi) {
      form <- as.formula(paste0(
        vi, " ~ 1 + (1|Code) + (1|Treatment) + ",
        "(1|Block_Treatment) + (1|Code:Treatment)"
      ))
      fit <- tryCatch(
        suppressWarnings(lmer(form, data = data_subset, REML = TRUE, control = ctrl)), 
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(VI = vi, Dataset = dataset_label, Condition = as.character(loc), H2 = NA))
      
      # Check for convergence issues
      if (!is.null(fit@optinfo$conv$lme4$code) && fit@optinfo$conv$lme4$code != 0) {
        return(tibble(VI = vi, Dataset = dataset_label, Condition = as.character(loc), H2 = NA))
      }
      
      vc <- as.data.frame(VarCorr(fit))
      resid_var <- attr(VarCorr(fit), "sc")^2
      get_v <- function(name) ifelse(name %in% vc$grp, vc$vcov[vc$grp == name], 0)
      vg <- get_v("Code")
      vgE <- get_v("Code:Treatment")
      
      nE <- n_distinct(data_subset$Treatment)
      nrep <- mean(table(data_subset$Code, data_subset$Treatment) > 0)
      
      denom <- vg + vgE/nE + resid_var/(nE*nrep)
      h2 <- ifelse(denom > 0, vg/denom, NA)
      
      tibble(VI = vi, Dataset = dataset_label, Condition = as.character(loc), H2 = h2)
    })
  })
  
  # === H² PER TREATMENT ===
  treatments <- unique(df$Treatment)
  results_all$by_treatment <- map_dfr(treatments, function(trt) {
    data_subset <- df %>% filter(Treatment == trt)
    
    map_dfr(vi_cols, function(vi) {
      form <- as.formula(paste0(
        vi, " ~ 1 + (1|Code) + (1|Location) + ",
        "(1|Block_Location) + (1|Code:Location)"
      ))
      fit <- tryCatch(
        suppressWarnings(lmer(form, data = data_subset, REML = TRUE, control = ctrl)), 
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(VI = vi, Dataset = dataset_label, Condition = as.character(trt), H2 = NA))
      
      # Check for convergence issues
      if (!is.null(fit@optinfo$conv$lme4$code) && fit@optinfo$conv$lme4$code != 0) {
        return(tibble(VI = vi, Dataset = dataset_label, Condition = as.character(trt), H2 = NA))
      }
      
      vc <- as.data.frame(VarCorr(fit))
      resid_var <- attr(VarCorr(fit), "sc")^2
      get_v <- function(name) ifelse(name %in% vc$grp, vc$vcov[vc$grp == name], 0)
      vg <- get_v("Code")
      vgL <- get_v("Code:Location")
      
      nL <- n_distinct(data_subset$Location)
      nrep <- mean(table(data_subset$Code, data_subset$Location) > 0)
      
      denom <- vg + vgL/nL + resid_var/(nL*nrep)
      h2 <- ifelse(denom > 0, vg/denom, NA)
      
      tibble(VI = vi, Dataset = dataset_label, Condition = as.character(trt), H2 = h2)
    })
  })
  
  # === Repeatability PER ENVIRONMENT (Location × Treatment) === within-trial broad-sense heritability
  environments <- unique(df$Environment)
  results_all$by_environment <- map_dfr(environments, function(env) {
    data_subset <- df %>% filter(Environment == env)
    
    map_dfr(vi_cols, function(vi) {
      form <- as.formula(paste0(vi, " ~ 1 + (1|Code) + (1|Block)"))
      fit <- tryCatch(
        suppressWarnings(lmer(form, data = data_subset, REML = TRUE, control = ctrl)), 
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(VI = vi, Dataset = dataset_label, Condition = as.character(env), H2 = NA))
      
      # Check for convergence issues
      if (!is.null(fit@optinfo$conv$lme4$code) && fit@optinfo$conv$lme4$code != 0) {
        return(tibble(VI = vi, Dataset = dataset_label, Condition = as.character(env), H2 = NA))
      }
      
      vc <- as.data.frame(VarCorr(fit))
      resid_var <- attr(VarCorr(fit), "sc")^2
      get_v <- function(name) ifelse(name %in% vc$grp, vc$vcov[vc$grp == name], 0)
      vg <- get_v("Code")
      
      nrep <- n_distinct(data_subset$Block)

      denom <- vg + resid_var/nrep
      h2 <- ifelse(denom > 0, vg/denom, NA)
      
      tibble(VI = vi, Dataset = dataset_label, Condition = as.character(env), H2 = h2)
    })
  })
  
  # Combine all results
  bind_rows(results_all)
}

# ===============================================================
# 5. RUN HERITABILITY ANALYSIS
# ===============================================================

res_an   <- calculate_heritability(df_an, "Anthesis")
res_gf   <- calculate_heritability(df_gf, "GrainFilling")
res_diff <- calculate_heritability(df_diff, "Diff")

herit_table <- bind_rows(res_an, res_gf, res_diff)

# --- Save results ---
write_csv(herit_table, "results/heritability_summary_all_conditions.csv")

# ===============================================================
# VISUALIZATION — Repeatability & Broad-sense H² 
# ===============================================================

library(tidyverse)
library(fmsb)
library(RColorBrewer)

# --- Prepare data ---
herit_table <- herit_table %>%
  mutate(
    Dataset = recode(Dataset, "Diff" = "Index-difference"),
    Condition = str_replace_all(Condition, "_", " "),
    Dataset = factor(Dataset, levels = c("Anthesis", "GrainFilling", "Index-difference"))
  )

# --- Define groupings ---
environments <- c("Aranjuez Rainfed", "Aranjuez Irrigated", "Valladolid Rainfed", "Valladolid Irrigated")
locations <- c("Aranjuez", "Valladolid")
treatments <- c("Irrigated", "Rainfed")
overall <- c("Overall")

# --- Color-blind friendly palette (Okabe-Ito) ---
cb_colors <- c("Anthesis"="#E69F00", "GrainFilling"="#56B4E9", "Index-difference"="#009E73")

# --- Radar plotting function ---
plot_radar <- function(conds, title_text, layout_matrix=NULL, filename, max_scale=1, min_scale=0, is_environment=FALSE){
  VI_order <- sort(unique(herit_table$VI))
  n_plots <- length(conds)
  
  # Determine main title
  if(is_environment){
    main_title <- "Repeatability (within trial heritability)"
  } else {
    main_title <- expression(paste("Broad sense heritability ", H^2))
  }
  
  png(filename, width = 3600, height = ifelse(n_plots>2, 4000, 2600), res = 300)
  
  if(!is.null(layout_matrix)){
    layout(layout_matrix)
  } else {
    par(mfrow=c(1,n_plots))
  }
  
  # Add more outer margin at the bottom for the legend space
  if(n_plots > 1){
    par(oma=c(2,4,5,4), mar=c(3,3,3,3))
    centerzero_val <- FALSE
  } else {
    par(oma=c(7,8,5,8), mar=c(6,6,5,6))
    centerzero_val <- TRUE
  }
  
  # --- Plot each condition ---
  for(cond in conds){
    df_cond <- herit_table %>% filter(Condition==cond) %>%
      group_by(Dataset, VI) %>% summarise(H2=mean(H2, na.rm=TRUE), .groups="drop") %>%
      pivot_wider(names_from=VI, values_from=H2)
    
    if(nrow(df_cond)==0) next
    df_cond <- df_cond %>% select(Dataset, all_of(VI_order)) %>% replace(is.na(.),0)
    
    radar_data <- rbind(
      rep(max_scale,length(VI_order)),
      rep(min_scale,length(VI_order)),
      df_cond[,-1]
    ) %>% as.data.frame()
    
    rownames(radar_data) <- c("Max","Min",as.character(df_cond$Dataset))
    
    n_vars <- length(VI_order)
    angles <- seq(pi/2, pi/2 - 2*pi, length.out = n_vars + 1)[1:n_vars]
    angles_deg <- angles * 180 / pi
    
    fmsb::radarchart(
      radar_data,
      axistype=1,
      pcol=cb_colors[df_cond$Dataset],
      plwd=2,
      plty=1,
      cglcol="grey70",
      cglty=1,
      cglwd=0.8,
      axislabcol="grey30",
      caxislabels=c("0","0.2","0.4","0.6","0.8","1.0"),
      vlabels=rep("", n_vars),
      title="",
      cex.main=1.2,
      centerzero=centerzero_val
    )
    
    title(cond, line=2, cex.main=1.4)
    
    # Tilted VI labels
    for(i in 1:n_vars){
      angle_rad <- angles[i]
      label_dist <- 1.15
      x <- label_dist * cos(angle_rad)
      y <- label_dist * sin(angle_rad)
      
      text_angle <- angles_deg[i]
      if(text_angle < -90 || text_angle > 90){
        text_angle <- text_angle + 180
        adj_val <- 1
      } else {
        adj_val <- 0
      }
      text(x, y, gsub("_", " ", VI_order[i]), srt=text_angle, adj=c(adj_val, 0.5), cex=0.85, xpd=TRUE)
    }
  }
  
  # --- Title ---
  mtext(main_title, side=3, outer=TRUE, cex=1.8, font=2, line=1.5)
  
  # --- Legend BELOW all plots ---
  if(n_plots > 2){
    par(fig=c(0,1,0,0.02), new=TRUE, mar=c(0,0,0,0))  # move legend further down
  } else {
    par(fig=c(0,1,0,0.10), new=TRUE, mar=c(0,0,0,0))
  }
  plot(0,0,type="n",axes=FALSE,xlab="",ylab="")
  legend("center", legend=names(cb_colors), col=cb_colors, lty=1, lwd=2,
         horiz=TRUE, bty="n", cex=1.2, y.intersp=1.2, x.intersp=0.8)
  
  dev.off()
}

# --- PLOT ENVIRONMENTS (2x2) - Repeatability ---
plot_radar(environments, "Environments",
           layout_matrix=matrix(1:4,2,2,byrow=TRUE),
           filename="radar_environments.png",
           is_environment=TRUE)



# # --- PLOT LOCATIONS (1x2) - H² ---
# plot_radar(locations, "Locations", 
#            layout_matrix=matrix(1:2,1,2,byrow=TRUE), 
#            filename="radar_locations.png",
#            is_environment=FALSE)
# 
# # --- PLOT TREATMENTS (1x2) - H² ---
# plot_radar(treatments, "Treatments", 
#            layout_matrix=matrix(1:2,1,2,byrow=TRUE), 
#            filename="radar_treatments.png",
#            is_environment=FALSE)
# 
# # --- PLOT OVERALL (single) - H² ---
# plot_radar(overall, "Overall", 
#            filename="radar_overall.png",
#            is_environment=FALSE)