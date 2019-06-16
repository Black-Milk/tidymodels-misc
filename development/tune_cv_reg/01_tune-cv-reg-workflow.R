# AUTOMATED PROCESS TO TUNE A PREASSEMBLED HYPERPARAMETER GRID USING K-FOLD
# CROSS-VALIDATION FOR PARSNIP REGRESSION MODELS


# Overall modeling flow for cross-validation
# 1. Split original data set into a training set and a testing set (e.g. 80%/20% split)
# 2. Split training set into k-folds
#         A fold is basically just a group (e.g. if k = 10, split training data into 10 groups)
#         Note that if we do k-fold cross-validation repeated n times, we will split into k-groups n-times
# For each fold:
#     3. Train the model on k-1 folds
#     4. Validate the model on the remaining fold (e.g. if regression model, can calculate RMSE)
#     5. Record the value of the validation metric
# 6. Average the validation metric across each fold
# 7. Average the averaged validation metric across all runs, if n > 1
# 8. Final validation metric for the model is the one calculated in Step 7
# 9. Note that the testing set would be used to calculate performance metric between different model types (e.g. logistic regression vs random forest)
#
# For using k-fold cross-validation to optimize hyperparameters for a model
# 1. Repeat Steps 1-8 for each parameter or combination of parameters attempted
# 2. Select optimal parameters based on highest or lowest validaiton metric (e.g. if r-squared, highest; if RMSE, lowest)
#
# High-level Example:
# 1. Start with a dataset of 1000 observations, and do 5-fold cross-validation
# 2. Split dataset into 80/20 training and testing sets
#       Training set would have 800 observations, testing set would have 200
# 3. Split training dataset into 5 folds of equal size
#       Each fold would have 160 observations, label these folds 1, 2, 3, 4, and 5
# For each fold 1, 2, 3, 4, and 5
#       4. Train model on folds 1-4 (analysis dataset), calculate metric (e.g. RMSE) using fold 5 (assessment dataset)
#       5. Train model on folds 2-5, calculate metric using fold 1
#       6. Train model on folds 1, 3-5, calculate metric using fold 2
#       7. Train model on folds 1, 2, 4, 5, calculate metrc using fold 3
#       8. Train model on folds 1-3, 5, calculate metric using fold 4
# 9. Average the five calculated metrics to get a single metric for this model with a particular set of hyperparameters


# 0. Setup ----


# 0.1 Packages ----

library(tidyverse)
library(glue)
library(tictoc)
library(tidymodels)


# 0.2 Data ----

# Load Data
regress_data_tbl <- parsnip::check_times %>%
  select(check_time, imports:Roxygen, r_count:s3_methods, src_count, data_count)

regress_data_tbl

# Set seed for reproducibility
set.seed(1234)

# Split dataset into training (80%) and testing (20%) datasets
split_obj <- initial_split(regress_data_tbl, prop = 0.8)
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)


# 1. Function Workflow ----

# Goal: a way to automate tuning a hyperparameter grid using k-fold cross-validation
#       (repeated n times, if desired)

# Construct an "un-trained" parsnip model with varying() parameters
model_base_obj <- parsnip::rand_forest(mode = "regression") %>%
  set_engine("randomForest") %>%
  set_args(mtry = varying(), min_n = varying(), trees = 100)

model_base_obj

# Construct a hyperparameter grid from the dials package
tuning_grid_tbl <- grid_random(
  range_set(mtry, c(1, 5)),
  range_set(min_n, c(2, 15)),
  size = 3
)

tuning_grid_tbl

# Merge the un-trained model base with the hyperparameter grid to get a list of "un-trained" models
merge(model_base_obj, tuning_grid_tbl)

# Use rsample package to create k folds, then use those folds to finish training the model
cv_obj <- vfold_cv(train_tbl, v = 3, repeats = 1)

cv_obj
cv_obj$id[[1]]
cv_obj$splits[[1]]


# There is a wide selection of metrics that can be used to validate a model
# Many of these metrics are functions in the yardstick package
# The yardstick package also provides a way to run multiple metric functions at once
# Here, use root mean squared error, r-squared, and mean absolute error
regress_metrics <- metric_set(rmse, rsq, mae)


# 1.1 Run Folds ----

# Step 1: Function that works for a single fold

# Use the first model/parameter combination
current_model <- merge(model_base_obj, tuning_grid_tbl)[[1]]
current_model

single_fold_result_tbl <- cv_obj %>%

  # Get the first fold
  slice(1) %>%

  # Use rsample splits to create analysis and assessment data
  mutate(
    analysis_tbl   = map(splits, analysis),
    assessment_tbl = map(splits, assessment)
  ) %>%

  # Train the model on the analysis data
  mutate(
    model_trained_obj = map(
      analysis_tbl,

      ~ fit(current_model, check_time ~ ., data = .x)
    )
  ) %>%

  # Calculate predictions using the assessment data
  mutate(
    predictions_tbl = map2(
      assessment_tbl,
      model_trained_obj,

      ~ tibble(
          truth    = pull(.x, check_time),
          estimate = predict(.y, new_data = .x)$.pred
      )
    )
  ) %>%

  # Calculate performance metrics
  mutate(
    metrics_tbl = map(
      predictions_tbl,

      ~ regress_metrics(data = .x, truth = truth, estimate = estimate)
    )
  )

single_fold_result_tbl

single_fold_result_tbl %>%
  unnest(metrics_tbl)


# Rewrite anonymous functions as named functions
predict_reg <- function(object, new_data, response_expr) {

  output_tbl <- tibble::tibble(
    truth    = dplyr::pull(new_data, !!response_expr),
    estimate = predict(object, new_data = new_data)$.pred
  )

}

fit_partial             <- purrr::partial(parsnip::fit, object = current_model, formula = check_time ~ .)
predict_reg_partial     <- purrr::partial(predict_reg, response_expr = quo(check_time))
regress_metrics_partial <- purrr::partial(regress_metrics, truth = truth, estimate = estimate)


single_fold_result_2_tbl <- cv_obj %>%

  slice(1) %>%

  mutate(
    analysis_tbl      = map(splits, analysis),
    assessment_tbl    = map(splits, assessment),
    model_trained_obj = map(analysis_tbl, fit_partial),
    predictions_tbl   = map2(model_trained_obj, assessment_tbl, predict_reg_partial),
    metrics_tbl       = map(predictions_tbl, regress_metrics_partial)
  )

single_fold_result_2_tbl

single_fold_result_2_tbl %>%
  unnest(metrics_tbl)

# Step 2: Function that works across multiple folds
multiple_folds_result_tbl <- cv_obj %>%
  mutate(
    analysis_tbl      = map(splits, analysis),
    assessment_tbl    = map(splits, assessment),
    model_trained_obj = map(analysis_tbl, fit_partial),
    predictions_tbl   = map2(model_trained_obj, assessment_tbl, predict_reg_partial),
    metrics_tbl       = map(predictions_tbl, regress_metrics_partial)
  )

multiple_folds_result_tbl

multiple_folds_result_tbl %>%
  unnest(metrics_tbl)

multiple_folds_result_tbl %>%
  unnest(metrics_tbl) %>%
  group_by(.metric) %>%
  summarize(.mean = mean(.estimate))


# 1.2 Run Repeats ----

# rsample allows repeated cross-validation... can apply the previous across all folds for all repeats
cv_2_obj <- vfold_cv(train_tbl, v = 3, repeats = 3)
cv_2_obj

multiple_runs_result_tbl <- cv_2_obj %>%
  mutate(
    analysis_tbl      = map(splits, analysis),
    assessment_tbl    = map(splits, assessment),
    model_trained_obj = map(analysis_tbl, fit_partial),
    predictions_tbl   = map2(model_trained_obj, assessment_tbl, predict_reg_partial),
    metrics_tbl       = map(predictions_tbl, regress_metrics_partial)
  )

multiple_runs_result_tbl

multiple_runs_result_tbl %>%
  unnest(metrics_tbl)

multiple_runs_result_tbl %>%
  unnest(metrics_tbl) %>%
  group_by(id, .metric) %>%
  summarize(.mean = mean(.estimate)) %>%
  ungroup()

multiple_runs_result_tbl %>%
  unnest(metrics_tbl) %>%
  group_by(.metric) %>%
  summarize(.mean = mean(.estimate)) %>%
  ungroup()


# 1.3 Run Parameters ----

# Running multiple parameters over multiple repeats can be time-consuming
# Add a message to the fit_model function to let user know each step of the process
fit_model <- function(object, data, formula, iter, run, fold) {

  start_time <- tictoc::tic(quiet = TRUE)

  output_obj <- object %>%
    parsnip::fit(formula, data = data)

  end_time   <- tictoc::toc(quiet = TRUE)
  total_time <- as.numeric(end_time$toc - end_time$tic) %>% round(1)

  print(glue::glue("Finished Combination {iter}, {run}, {fold} (time: {total_time} s)"))

  return(output_obj)

}

# Step 1: Function that works for a single parameter iteration

# Define constants for first iteration (model/parameter combination and formula)
current_combo <- slice(tuning_grid_tbl, 1)
current_model <- merge(model_base_obj, current_combo)[[1]]
model_formula <- as.formula("check_time ~ .")

current_combo
current_model
model_formula

cv_2_obj

# Partialize functions (pre-load each function with the arguments that do not change)
fit_model_partial       <- purrr::partial(fit_model, object = current_model, formula = model_formula, iter = 1)
predict_reg_partial     <- purrr::partial(predict_reg, response_expr = quo(check_time))
regress_metrics_partial <- purrr::partial(regress_metrics, truth = truth, estimate = estimate)


single_iteration_result_tbl <- cv_2_obj %>%
  mutate(
    analysis_tbl      = map(splits, analysis),
    assessment_tbl    = map(splits, assessment),
    model_trained_obj = pmap(list(data = analysis_tbl, run = id, fold = id2), fit_model_partial),
    predictions_tbl   = map2(model_trained_obj, assessment_tbl, predict_reg_partial),
    metrics_tbl       = map(predictions_tbl, regress_metrics_partial)
  )

single_iteration_result_tbl

single_iteration_result_tbl %>%
  unnest(metrics_tbl)

single_iteration_result_tbl %>%
  unnest(metrics_tbl) %>%
  group_by(id, .metric) %>%
  summarize(.mean = mean(.estimate)) %>%
  ungroup()

single_iteration_result_tbl %>%
  unnest(metrics_tbl) %>%
  group_by(.metric) %>%
  summarize(.mean = mean(.estimate)) %>%
  ungroup()

# Function to apply all of the above to a list of un-trained parsnip models
run_parameters <- function(object, iteration, cv, response_expr) {

  model_formula <- as.formula(glue::glue("{quo_name(response_expr)} ~ ."))

  fit_model_partial       <- purrr::partial(fit_model, object = object, formula = model_formula)
  predict_reg_partial     <- purrr::partial(predict_reg, response_expr = response_expr)
  regress_metrics_partial <- purrr::partial(regress_metrics, truth = truth, estimate = estimate)

  output_tbl <- cv %>%
    mutate(
      analysis_tbl      = map(splits, analysis),
      assessment_tbl    = map(splits, assessment),
      model_trained_obj = pmap(list(data = analysis_tbl, iter = iteration, run = id, fold = id2), fit_model_partial),
      predictions_tbl   = map2(model_trained_obj, assessment_tbl, predict_reg_partial),
      metrics_tbl       = map(predictions_tbl, regress_metrics_partial)
    )

  return(output_tbl)

}

run_parameters_partial <- purrr::partial(run_parameters, cv = cv_2_obj, response_expr = quo(check_time))

single_iteration_result_2_list <- list(current_model) %>%
  imap(run_parameters_partial)

single_iteration_result_2_list

single_iteration_result_2_list %>%
  pluck(1) %>%
  unnest(metrics_tbl)

single_iteration_result_2_list %>%
  pluck(1) %>%
  unnest(metrics_tbl) %>%
  group_by(id, .metric) %>%
  summarize(.mean = mean(.estimate))

single_iteration_result_2_list %>%
  pluck(1) %>%
  unnest(metrics_tbl) %>%
  group_by(.metric) %>%
  summarize(.mean = mean(.estimate))


# Step 2: Apply function across multiple parameter iterations
all_models <- merge(model_base_obj, tuning_grid_tbl)
all_models

multiple_iteration_result_list <- all_models %>%
  imap(run_parameters_partial)

multiple_iteration_result_list

multiple_iteration_result_list %>%
  map(~ unnest(.x, metrics_tbl))

multiple_iteration_result_list %>%
  map(~ .x %>% unnest(metrics_tbl) %>% group_by(id, .metric) %>% summarize(.mean = mean(.estimate)))

multiple_iteration_result_list %>%
  map(~ .x %>% unnest(metrics_tbl) %>% group_by(.metric) %>% summarize(.mean = mean(.estimate)))

multiple_iteration_result_list %>%
  imap(
    ~ .x %>%
        unnest(metrics_tbl) %>%
        group_by(.metric) %>%
        summarize(.mean = mean(.estimate)) %>%
        mutate(combo = as.character(.y)) %>%
        spread(key = .metric, value = .mean)
    )

multiple_iteration_result_list %>%
  imap(
    ~ .x %>%
      unnest(metrics_tbl) %>%
      group_by(.metric) %>%
      summarize(.mean = mean(.estimate)) %>%
      mutate(combo = as.character(.y)) %>%
      spread(key = .metric, value = .mean)
  ) %>%
  reduce(bind_rows)


# Step 3: Combine results and append to original tuning grid
tuning_grid_results_tbl <- multiple_iteration_result_list %>%
  imap(
    ~ .x %>%
      unnest(metrics_tbl) %>%
      group_by(.metric) %>%
      summarize(.mean = mean(.estimate)) %>%
      mutate(combination = as.character(.y)) %>%
      spread(key = .metric, value = .mean)
  ) %>%
  reduce(bind_rows) %>%
  inner_join(rownames_to_column(tuning_grid_tbl, var = "combination")) %>%
  select(-combination) %>%
  select(one_of(names(tuning_grid_tbl)), rmse, rsq, mae)

tuning_grid_results_tbl


# Step 4: Train final model based on tuning results

# Will use rmse as the performance metric
tuning_grid_results_tbl %>%
  arrange(rmse) %>%
  slice(1)

# mtry = 2 and min_n = 10
final_model_obj <- model_base_obj %>%
  set_args(mtry = 2, min_n = 10) %>%
  fit(check_time ~ ., data = train_tbl)

final_model_obj


# 2. Write Function ----


# 2.1 Testing Values ----

object <- parsnip::rand_forest(mode = "regression") %>%
  set_engine("randomForest") %>%
  set_args(mtry = varying(), min_n = varying(), trees = 100)

data <- train_tbl

grid <- grid_random(
  range_set(mtry, c(1, 5)),
  range_set(min_n, c(2, 15)),
  size = 3
)

response_expr <- quo(check_time)

k        <- 3
n        <- 3
.verbose <- TRUE


# 2.2 Define Helper Functions ----

regress_metrics <- yardstick::metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)

fit_model <- function(object, data, formula, iter, run, fold, .verbose) {

  start_time <- tictoc::tic(quiet = TRUE)

  output_obj <- object %>%
    parsnip::fit(formula, data = data)

  end_time   <- tictoc::toc(quiet = TRUE)
  total_time <- as.numeric(end_time$toc - end_time$tic) %>% round(1)

  if (.verbose)
    print(glue::glue("Finished Combination {iter}, {run}, {fold} (time: {total_time} s)"))

  return(output_obj)

}

predict_reg <- function(object, new_data, response_expr) {

  output_tbl <- tibble::tibble(
    truth    = dplyr::pull(new_data, !!response_expr),
    estimate = predict(object, new_data = new_data)$.pred
  )

}

run_parameters <- function(object, iteration, cv, response_expr, .verbose) {

  model_formula <- as.formula(glue::glue("{quo_name(response_expr)} ~ ."))

  fit_model_partial       <- purrr::partial(fit_model, object = object, formula = model_formula, .verbose = .verbose)
  predict_reg_partial     <- purrr::partial(predict_reg, response_expr = response_expr)
  regress_metrics_partial <- purrr::partial(regress_metrics, truth = truth, estimate = estimate)

  output_tbl <- cv %>%
    mutate(
      analysis_tbl      = map(splits, analysis),
      assessment_tbl    = map(splits, assessment),
      model_trained_obj = pmap(list(data = analysis_tbl, iter = iteration, run = id, fold = id2), fit_model_partial),
      predictions_tbl   = map2(model_trained_obj, assessment_tbl, predict_reg_partial),
      metrics_tbl       = map(predictions_tbl, regress_metrics_partial)
    )

  return(output_tbl)

}

summarize_metrics <- function(results, index) {

  output_tbl <- results %>%
    tidyr::unnest(metrics_tbl) %>%
    dplyr::group_by(.metric) %>%
    dplyr::summarize(.mean = mean(.estimate)) %>%
    dplyr::mutate(combination = as.character(index)) %>%
    tidyr::spread(key = .metric, value = .mean)

  return(output_tbl)

}


# 2.3 Define Main Function ----

tune_cv_reg <- function(object, data, grid, response, k = 5, n = 3, .verbose = FALSE) {

  # Input Validation
  is_parsnip_obj <- inherits(object, "model_spec")
  if (!is_parsnip_obj)
    stop("Argument 'object' must be a parsnip package model object")

  is_df_or_tbl <- inherits(data, "data.frame") | inherits(data, "tbl")
  if (!is_df_or_tbl)
    stop("Argument 'data' must be a data frame or tibble")

  is_dials_grid <- inherits(grid, "param_grid")
  if (!is_dials_grid)
    stop("Argument 'grid' must be a dials package grid_random or grid_regular")

  unique_grid_combos <- grid %>% dplyr::group_by_all() %>% dplyr::distinct() %>% nrow() == nrow(grid)
  if (!unique_grid_combos)
    stop("Argument 'grid' must have all unique rows (parameter combinations must be unique)")

  is_valid_quosure <- quo_name(enquo(response)) %in% names(data)
  if (!is_valid_quosure)
    stop("Argument 'response' must be an unquoted valid column name from the 'data' argument")

  is_continuous_outcome <- inherits(dplyr::pull(data, !!enquo(response)), "numeric")
  if (!is_continuous_outcome)
    stop("This function is for regressions, argument 'response' must reference a variable of type 'numeric'")

  k_int             <- as.integer(k)
  n_int             <- as.integer(n)
  are_positive_ints <- inherits(k_int, "integer") & inherits(n_int, "integer") & k_int > 0 & n_int > 0
  if (!are_positive_ints)
    stop("Arguments 'k' and 'n' must be of type integer and both be greater than zero")

  is_logic_arg <- inherits(.verbose, "logical")
  if (!is_logic_arg)
    stop("Argument '.verbose' must be of type ")


  # Input formatting
  response_expr    <- dplyr::enquo(response)
  num_param_combos <- nrow(grid)
  iteration_count  <- 1:num_param_combos
  model_obj_list   <- merge(object, grid)
  model_formula    <- as.formula(glue::glue("{quo_name(response_expr)} ~ ."))
  cv_obj           <- rsample::vfold_cv(data, v = k_int, repeats = n_int)

  if (n == 1)
    cv_obj %>% dplyr::rename(id2 = id) %>% dplyr::mutate(id = "Repeat1")


  # Partialize Functions
  fit_model_partial       <- purrr::partial(fit_model, object = object, formula = model_formula, .verbose = .verbose)
  predict_reg_partial     <- purrr::partial(predict_reg, response_expr = response_expr)
  run_parameters_partial  <- purrr::partial(run_parameters, cv = cv_obj, response_expr = response_expr, .verbose = .verbose)
  regress_metrics_partial <- purrr::partial(regress_metrics, truth = truth, estimate = estimate)


  # Run Functions
  output_tbl <- model_obj_list %>%
    purrr::imap(run_parameters_partial) %>%
    purrr::imap(summarize_metrics) %>%
    purrr::reduce(dplyr::bind_rows) %>%
    dplyr::inner_join(tibble::rownames_to_column(grid, var = "combination"), by = "combination") %>%
    dplyr::select(-combination) %>%
    dplyr::select(dplyr::one_of(names(grid)), rmse, rsq, mae)


  return(output_tbl)

}


tuned_grid_tbl <- object %>%
  tune_cv_reg(
    data     = data,
    grid     = grid,
    response = check_time,
    k        = 3,
    n        = 2,
    .verbose = TRUE
  )

tuned_grid_tbl

tuned_grid_tbl %>%
  arrange(rmse) %>%
  slice(1)

final_model_obj <- object %>%
  set_args(mtry = 4, min_n = 14) %>%
  fit(check_time ~ ., data = train_tbl)

final_model_obj
