# TESTING tune_cv_reg() ----

# 0. Setup ----


# 0.1 Packages ----

library(tidyverse)
library(tidymodels)


# 0.2 Data ----

data("check_times")
data("diamonds")
data("mpg")
data("midwest")

check_times_tbl <- check_times %>%
  select(check_time, authors:testthat_size)

diamonds_tbl <- diamonds %>%
  select(price, everything()) %>%
  mutate(price = as.numeric(price))

mpg_tbl <- mpg %>%
  select(hwy, manufacturer, displ:drv, fl, class) %>%
  mutate_if(is.character, as_factor) %>%
  mutate(cyl = as.factor(cyl)) %>%
  mutate(hwy = as.numeric(hwy))

midwest_tbl <- midwest %>%
  select(area, state, popdensity, contains("perc"), inmetro)

rm(check_times, diamonds, mpg, midwest)


# 0.3 Functions ----

source(here::here("functions/tune_cv_reg.R"))


# 1. Run Tests ----


# 1.1 Test 1 ----

# Dataset: check_times
# Model: Multivariate Adaptive Regression Splines (mars)
# Parameters: num_terms

set.seed(1234)

model_1_base_obj <- mars(mode = "regression") %>%
  set_engine("earth") %>%
  set_args(num_terms = varying(), prod_degree = 1, prune_method = "none")
model_1_base_obj

grid_tbl <- grid_regular(range_set(num_terms, c(2, 5)), levels = 4)
grid_tbl

check_times_tbl

split_obj             <- initial_split(check_times_tbl, prob = 0.8)
check_times_train_tbl <- training(split_obj)
check_times_test_tbl  <- testing(split_obj)
check_times_train_tbl

model_1_params_tbl <- model_1_base_obj %>%
  tune_cv_reg(
    data     = check_times_train_tbl,
    grid     = grid_tbl,
    response = check_time,
    k        = 3,
    n        = 2,
    .verbose = TRUE
  )

model_1_params_tbl

model_1_params_tbl %>%
  arrange(desc(rsq)) %>%
  slice(1) # 5

model_1_trained_obj <- model_1_base_obj %>%
  update(num_terms = 5) %>%
  fit(check_time ~ ., data = check_times_train_tbl)

model_1_trained_obj


# 1.2 Test 2 ----

# Dataset: diamonds
# Model: Random Forest (ranger)
# Parameters: mtry, min_n

set.seed(1234)

model_2_base_obj <- rand_forest(mode = "regression") %>%
  set_engine("ranger") %>%
  set_args(trees = 100, mtry = varying(), min_n = varying())
model_2_base_obj

grid_tbl <- grid_random(
  mtry %>% range_set(c(2, 5)),
  min_n %>% range_set(c(2, 5)),
  size = 4
)
grid_tbl

diamonds_tbl

split_obj          <- initial_split(diamonds_tbl, prob = 0.8)
diamonds_train_tbl <- training(split_obj)
diamonds_test_tbl  <- testing(split_obj)
diamonds_train_tbl

model_2_params_tbl <- model_2_base_obj %>%
  tune_cv_reg(
    data     = diamonds_train_tbl,
    grid     = grid_tbl,
    response = price,
    k        = 3,
    n        = 2,
    .verbose = TRUE
  )

model_2_params_tbl

model_2_params_tbl %>%
  arrange(mae) %>%
  slice(1)  # mtry = 5, min_n = 5

model_2_trained_obj <- model_2_base_obj %>%
  update(mtry = 5, min_n = 5) %>%
  fit(price ~ ., data = diamonds_train_tbl)

model_2_trained_obj


# 1.3 Test 3 ----

# Dataset: mpg
# Model: Multilayer Perceptron (mlp) (nnet)
# Parameters: hidden_units, penalty, epochs

set.seed(1234)

model_3_base_obj <- mlp(mode = "regression") %>%
  set_engine("nnet") %>%
  set_args(hidden_units = varying(), penalty = varying(), epochs = varying())
model_3_base_obj

grid_tbl <- grid_random(
  hidden_units %>% range_set(c(9, 10)),
  penalty %>% range_set(c(0.02, 0)),
  epochs,
  size = 5
)
grid_tbl

mpg_tbl

split_obj     <- initial_split(mpg_tbl, prob = 0.8)
mpg_train_tbl <- training(split_obj)
mpg_test_tbl  <- testing(split_obj)
mpg_train_tbl

model_3_params_tbl <- model_3_base_obj %>%
  tune_cv_reg(
    data     = mpg_train_tbl,
    grid     = grid_tbl,
    response = hwy,
    k        = 3,
    n        = 2,
    .verbose = TRUE
  )

model_3_params_tbl

model_3_params_tbl %>%
  arrange(rmse) %>%
  slice(1)  # hidden_units = 9, penalty = 1.02, epochs = 726

model_3_trained_obj <- model_3_base_obj %>%
  update(hidden_units = 9, penalty = 1.02, epochs = 726) %>%
  fit(hwy ~ ., data = mpg_train_tbl)

model_3_trained_obj


# 1.3 Test 4 ----

# Dataset: midwest
# Model: boosted tree (xgboost)
# Parameters: mtry, min_n, tree_depth, learn_rate, loss_reduction, sample_size

set.seed(1234)

model_4_base_obj <- boost_tree(mode = "regression") %>%
  set_engine("xgboost") %>%
  set_args(trees = 100, mtry = varying(), min_n = varying(), tree_depth = varying(),
           learn_rate = varying(), loss_reduction = varying(), sample_size = varying())
model_4_base_obj

grid_tbl <- grid_random(
  mtry %>% range_set(c(5, 16)),
  min_n %>% range_set(c(15, 30)),
  tree_depth,
  learn_rate %>% range_set(c(0.05, 0.3)),
  loss_reduction %>% range_set(c(0.1, 0.2)),
  sample_size %>% range_set(c(0.5, 0.75)),
  size = 5
)
grid_tbl

mpg_tbl

split_obj     <- initial_split(midwest_tbl, prob = 0.8)
midwest_train_tbl <- training(split_obj)
midwest_test_tbl  <- testing(split_obj)
midwest_train_tbl

model_4_params_tbl <- model_4_base_obj %>%
  tune_cv_reg(
    data     = midwest_train_tbl,
    grid     = grid_tbl,
    response = area,
    k        = 3,
    n        = 2,
    .verbose = TRUE
  )

model_4_params_tbl

model_4_params_tbl %>%
  arrange(rmse) %>%
  slice(1)
  # mtry = 10, min_n = 15, tree_depth = 3, learn_rate = 0.278
  # loss_reduction = 0.112, sample_size = 0.5

model_4_trained_obj <- model_4_base_obj %>%
  update(mtry = 10, min_n = 15, tree_depth = 3, learn_rate = 0.278,
         loss_reduction = 0.122, sample_size = 0.5) %>%
  fit(area ~ ., data = midwest_train_tbl)

model_4_trained_obj
