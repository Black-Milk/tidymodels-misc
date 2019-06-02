# STANDARDIZED OUTPUT FOR BINARY CLASSIFICATION MODEL PREDICTION OUTPUT

# 0. Setup ----


# 0.1 Packages ----

library(tidyverse)
library(glue)
library(tidymodels)


# 0.2 Data ----

class_data_tbl <- parsnip::lending_club %>%
  select(Class, funded_amnt:sub_grade, annual_inc, emp_length, revol_util)

class_data_tbl


split_obj <- initial_split(class_data_tbl, prop = 0.8, strata = "Class")
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)


# 0.3 Models ----

model_logit1_obj <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(Class ~ ., data = train_tbl)

model_logit2_obj <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit_xy(x = select(train_tbl, -Class), y = select(train_tbl, Class))


model_rf1_obj <- rand_forest(mode = "classification") %>%
  set_engine("randomForest") %>%
  set_args(mtry = 4, trees = 50) %>%
  fit(Class ~ ., data = train_tbl)

model_rf2_obj <- rand_forest(mode = "classification") %>%
  set_engine("randomForest") %>%
  set_args(mtry = 4, trees = 50) %>%
  fit_xy(x = select(train_tbl, -Class), y = select(train_tbl, Class))

model_lm_obj <- linear_reg(mode = "regression") %>%
  set_engine("lm") %>%
  fit(annual_inc ~ ., data = train_tbl)


# 1. Function Workflow ----

# Goal: simple function to provide output for binary classification predictions
# Truth: actual outcomes from dataset
# "Class1": probability of the first class
# "Class2": probability of the second class
# Prediction: the predicted class

# Step 1: Determine 'truth' column (just the response variable)

test_tbl %>%
  select(Class)

# Notes: will need a variable for the data, and a quosure for the response variable


# Step 2: Calculate probabilities

model_logit1_obj %>%
  predict(new_data = test_tbl, type = "prob")

# Notes: will need a variable for model and type (prob)
# NOtes: will want to change column names and remove '.pred_'


# Step 3: Determine outcome classes

model_logit1_obj %>%
  predict(new_data = test_tbl)

# Will want to add some control over this step by allowing a threshold
# Notes: will want a .thresh argument, set to 0.5 as a default

model_logit1_obj %>%
  predict(new_data = test_tbl, type = "prob") %>%
  set_names(str_remove_all(names(.), ".pred_")) %>%
  select(good) %>%
  mutate(
    prediction = case_when(
      good >= 0.2 ~ "good",
      TRUE        ~ "bad"
    )
  ) %>%
  select(prediction)

# Notes: will need to infer which response level is positive and which is negative
# Notes: will want to change column names to something generic so can handle any column name
# Notes: make sure to set .thresh at 0.5 for default
# Notes: if .thresh is changed, make sure it is between 0 and 1


# 2. Write Function ----

object   <- model_logit_obj
new_data <- test_tbl
response <- quo(Class)
positive <- "good"
.thresh  <- 0.3
.verbose <- TRUE

predict_binary <- function(object, new_data, response, positive = NULL,
                           .thresh = 0.5, .verbose = FALSE) {

  # Format input variable(s) and preliminary check(s)
  mode <- object$spec$mode

  if (mode != "classification") stop("predict_binary() should only be used with classification models")

  response_expr <- enquo(response)

  response_is_factor <- new_data %>%
    pull(!!response_expr) %>%
    is.factor()

  if (!response_is_factor) stop("Response variable must be a factor")

  outcome_levels <- new_data %>% pull(!!response_expr) %>% levels()

  if (!between(.thresh, 0, 1)) stop("Threshold must be between 0 and 1")


  # Main function body

  # Determine truth
  truth_tbl <- new_data %>%
    select(!!response_expr) %>%
    set_names("truth")

  if (.verbose) print(glue("Truth column set to {quo_name(response_expr)}..."))

  # Calculate probabilities
  probs_tbl <- object %>%
    predict(new_data = new_data, type = "prob") %>%
    set_names(str_remove_all(names(.), ".pred_"))

  if (.verbose) print(glue("Predicted probabilities calculated..."))

  # Determine positive outcome level
  if (is.null(positive)) {

    inferred_positive_level <- probs_tbl %>%
      gather() %>%
      group_by(key) %>%
      summarize(mean_prob = mean(value)) %>%
      ungroup() %>%
      filter(mean_prob == max(mean_prob)) %>%
      pull(key)

    if (.verbose)
      print(glue("No positive outcome level given... inferred as {inferred_positive_level}"))

  } else {

    if (positive %in% outcome_levels) {

      inferred_positive_level <- positive

    } else {

      stop(glue("Level {positive} is not a level in response variable {quo_name(response_expr)}"))

    }


  }

  # Calculate predicted class
  inferred_negative_level <- outcome_levels[outcome_levels != inferred_positive_level]

  preds_tbl <- probs_tbl %>%
    select(inferred_positive_level) %>%
    set_names("Class1") %>%
    mutate(
      prediction = case_when(
        Class1 >= .thresh ~ inferred_positive_level,
        TRUE              ~ inferred_negative_level
      ) %>% factor(levels = outcome_levels)
    ) %>%
    select(prediction)

  if (.verbose) print(glue("Predicted response classes determined..."))


  output_tbl <- truth_tbl %>%
    bind_cols(probs_tbl) %>%
    bind_cols(preds_tbl)

  if (.verbose) print(glue("Function completed successfully"))

  return(output_tbl)

}


# 3. Test Function ----

# Scenarios that should work
test1 <- model_logit1_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .verbose = TRUE)

test2 <- model_logit1_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .verbose = FALSE)


test3 <- model_logit2_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .verbose = TRUE)

test4 <- model_logit2_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .verbose = FALSE)


test5 <- model_rf1_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .verbose = TRUE)


# Scenarios that should fail
test6 <- model_rf1_obj %>%
  predict_binary(new_data = test_tbl, response = annual_inc)

test7 <- model_rf1_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "wrong", .verbose = TRUE)

test8 <- model_rf1_obj %>%
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .thresh = 1.2)

test9 <- model_rf1_obj %>%
  predict_binary(new_data = test_tbl, response = term, positive = "term_36")

test10 <- model_lm_obj %>%
  predict_binary(new_data = test_tbl, response = annual_inc)
