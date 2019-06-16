# TESTING predict_binary() ----

# 0. Setup ----


# 0.1 Packages ----

library(tidyverse)
library(tidymodels)


# 0.2 Data ----

data("attrition")
data("credit_data")

attrition_tbl <- attrition %>%
  as_tibble() %>%
  select(Attrition, Age, BusinessTravel, Department, Education, DailyRate,
         Gender, JobLevel, JobRole, PercentSalaryHike, WorkLifeBalance,
         YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion)

credit_data_tbl <- credit_data %>%
  as_tibble() %>%
  drop_na() %>%
  select(Status, Seniority, Home, Age, Marital, Job, Expenses,
         Income, Assets, Debt, Amount)


# 0.3 Function ----

source(here::here("functions/predict_binary.R"))


# 1. Run Tests ----


# 1.1 Test 1 ----

# Dataset: attrition
# Model: Logistic Regression

set.seed(1234)

split_1_obj         <- initial_split(attrition_tbl, prop = 0.8)
attrition_train_tbl <- training(split_1_obj)
attrition_test_tbl  <- testing(split_1_obj)

model_logit_obj <- logistic_reg() %>%
  set_engine("glm") %>%
  fit(Attrition ~ ., data = attrition_train_tbl)

tidy(model_logit_obj)

attrition_test_tbl %>%
  count(Attrition)
  # 40 / (40 + 253) = 0.14

results_1_tbl <- model_logit_obj %>%
  predict_binary(
    new_data = attrition_test_tbl,
    response = Attrition,
    positive = "Yes",
    .thresh  = 0.14,
    .verbose = TRUE
  )

results_1_tbl


# 1.2 Test 2 ----

# Dataset: credit_data
# Model: Random Forest

set.seed(1234)

split_2_obj           <- initial_split(credit_data_tbl, prop = 0.8)
credit_data_train_tbl <- training(split_2_obj)
credit_data_test_tbl  <- testing(split_2_obj)

model_rf_obj <- rand_forest(mode = "classification") %>%
  set_engine("randomForest") %>%
  set_args(mtry = 9, trees = 100) %>%
  fit(Status ~ ., data = credit_data_train_tbl)

credit_data_test_tbl %>%
  count(Status)
# 209 / (209 + 598) = 26%

results_2_tbl <- model_rf_obj %>%
  predict_binary(
    new_data = credit_data_test_tbl,
    response = Status,
    positive = "bad",
    .thresh  = 0.26,
    .verbose = TRUE
  )

results_2_tbl
