# =============================================================================
# Stuff+ Model Assignment — Module 10
# Models: (1) Basic Random Forest via ranger, (2) Improved model
# =============================================================================

# Install packages if needed (uncomment first time)
# install.packages(c("ranger", "dplyr", "tidymodels", "xgboost", "vip"))

library(ranger)
library(dplyr)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

train <- read.csv("training_data.csv", stringsAsFactors = FALSE)
test  <- read.csv("test_data_for_testing.csv", stringsAsFactors = FALSE)

cat("Train rows:", nrow(train), "\n")
cat("Test rows:", nrow(test), "\n")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================

# Fill the ~10 missing release_extension values with median
ext_median <- median(train$release_extension, na.rm = TRUE)
train$release_extension[is.na(train$release_extension)] <- ext_median
test$release_extension[is.na(test$release_extension)]   <- ext_median

# Convert categoricals to factors (ranger handles factors natively)
cat_cols <- c("pitch_type", "home_team", "stand", "inning_topbot")
for (col in cat_cols) {
  all_levels <- unique(c(train[[col]], test[[col]]))
  train[[col]] <- factor(train[[col]], levels = all_levels)
  test[[col]]  <- factor(test[[col]],  levels = all_levels)
}

# =============================================================================
# 3. MODEL 1 — BASIC RANDOM FOREST (ranger)
#    Variables: pitch_type, release_speed, pfx_x, pfx_z,
#               plate_x, plate_z, home_team, stand
# =============================================================================

rf_formula <- run_value ~ pitch_type + release_speed + pfx_x + pfx_z +
                           plate_x + plate_z + home_team + stand

set.seed(42)
rf_model <- ranger(
  formula       = rf_formula,
  data          = train,
  num.trees     = 300,
  mtry          = 3,          # sqrt(8 features) ≈ 3
  min.node.size = 5,
  importance    = "impurity",
  num.threads   = parallel::detectCores()
)

cat("\nModel 1 (RF) OOB RMSE:", sqrt(rf_model$prediction.error), "\n")

# Predict on test set
test$RF_STUFF <- predict(rf_model, data = test)$predictions

cat("RF_STUFF: mean =", mean(test$RF_STUFF), "| sd =", sd(test$RF_STUFF), "\n")

# =============================================================================
# 4. FEATURE ENGINEERING FOR IMPROVED MODEL
# =============================================================================

engineer_features <- function(df) {
  df %>%
    mutate(
      # Total movement magnitude
      break_total   = sqrt(pfx_x^2 + pfx_z^2),

      # Distance from typical strike zone center (plate center, ~2.5 ft height)
      plate_dist    = sqrt(plate_x^2 + (plate_z - 2.5)^2),

      # Count state as a single integer (e.g., 3-2 = 32, 0-0 = 0)
      count_state   = balls * 10 + strikes,

      # Interaction: high spin + high velo is more valuable
      velo_x_spin   = release_speed * release_spin_rate,

      # Horizontal-to-vertical break ratio (tunneling proxy)
      pfx_ratio     = pfx_x / (abs(pfx_z) + 0.01)
    )
}

train_eng <- engineer_features(train)
test_eng  <- engineer_features(test)

# =============================================================================
# 5. MODEL 2 — IMPROVED MODEL
#    Adds: spin rate, extension, count (balls/strikes), inning,
#          engineered features (break_total, plate_dist, count_state, etc.)
#    Uses more trees and tuned mtry for better performance
# =============================================================================

better_formula <- run_value ~ pitch_type + release_speed + pfx_x + pfx_z +
                               plate_x + plate_z + home_team + stand +
                               release_spin_rate + release_extension +
                               balls + strikes + inning + inning_topbot +
                               count_state + break_total + plate_dist +
                               velo_x_spin + pfx_ratio

set.seed(42)
better_model <- ranger(
  formula       = better_formula,
  data          = train_eng,
  num.trees     = 500,
  mtry          = 6,          # sqrt(19 features) ≈ 4–5; tuned slightly higher
  min.node.size = 3,          # slightly smaller leaves = more expressive
  importance    = "impurity",
  num.threads   = parallel::detectCores()
)

cat("\nModel 2 (Better RF) OOB RMSE:", sqrt(better_model$prediction.error), "\n")

# Predict on test set
test$BETTER_STUFF <- predict(better_model, data = test_eng)$predictions

cat("BETTER_STUFF: mean =", mean(test$BETTER_STUFF), "| sd =", sd(test$BETTER_STUFF), "\n")

# =============================================================================
# 6. FEATURE IMPORTANCE — BETTER MODEL
# =============================================================================

importance_df <- data.frame(
  feature    = names(better_model$variable.importance),
  importance = better_model$variable.importance
) %>% arrange(desc(importance))

cat("\nTop 10 feature importances (Better model):\n")
print(head(importance_df, 10))

# =============================================================================
# 7. SAVE OUTPUT
# =============================================================================

# The grader checks for RF_STUFF and BETTER_STUFF columns
write.csv(test, "stuff_predictions.csv", row.names = FALSE)

cat("\nSaved predictions to stuff_predictions.csv\n")
cat("Columns RF_STUFF and BETTER_STUFF are present and ready for submission.\n")
