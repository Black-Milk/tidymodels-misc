# STANDARDIZED OUTPUT FOR BINARY CLASSIFICATION MODEL PREDICTION OUTPUT

# 0. Setup ----


# 0.1 Packages ----

library(tidymodels)
library(tidyverse)
library(glue)


# 0.2 Data ----

set.seed(1234)

class_data_tbl <- parsnip::lending_club %>%
  select(Class, funded_amnt:sub_grade, annual_inc, emp_length, revol_util)

class_data_tbl

split_obj <- initial_split(class_data_tbl, prop = 0.8, strata = "Class")
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

train_tbl


# 0.3 Model ----

model_logit_obj <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(Class ~ ., data = train_tbl)

model_logit_obj
tidy(model_logit_obj)


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

model_logit_obj %>%
  predict(new_data = test_tbl, type = "prob")

# Notes: will need a variable for model and type (prob)
# NOtes: will want to change column names and remove '.pred_'


# Step 3: Determine outcome classes

model_logit_obj %>%
  predict(new_data = test_tbl)

# Will want to add some control over this step by allowing a threshold
# Notes: will want a .thresh argument, set to 0.5 as a default

model_logit_obj %>%
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

object        <- model_logit_obj
new_data      <- test_tbl
response_expr <- quo(Class)
positive      <- "good"
.thresh       <- 0.3
.verbose      <- TRUE

predict_binary <- function(object, new_data, response, positive = NULL,
                           .thresh = 0.5, .verbose = FALSE) {

  # Input Validation
  if (!inherits(object, "model_spec")) {
    stop("Argument 'object' must be a parsnip package model object")
  } else if (object$spec$mode != "classification") {
    stop("Argument 'obkect' must be a classification model")
  } else if (!inherits(data, c("data.frame", "tbl"))) {
    stop("Argument 'data' must be a data frame or tibble")
  } else if (!(quo_name(enquo(response)) %in% names(data))) {
    stop("Argument 'response' must be an unquoted valid column name from the 'data' argument")
  } else if (!inherits(dplyr::pull(data, !!enquo(response)), "factor")) {
    stop("This function is for classifications, argument 'response' must reference a variable of type 'factor'")
  } else if (length(levels(dplyr::pull(new_data, !!enquo(response)))) != 2) {
    stop("This function is for binary classifications, argument 'response' must have only two levels")
  } else if (!is.null(positive) & !(positive %in% levels(dplyr::pull(new_data, !!enquo(response))))) {
    stop("If non-NULL, argument 'positive' must match one of the factor levels in the column referenced by argument 'response'")
  } else if (!dplyr::between(.thresh, 0, 1)) {
    stop("Argument '.thresh' must be of type 'numeric' and must be between 0 and 1")
  } else if (!inherits(.verbose, "logical")) {
    stop("Argument '.verbose' must be of type 'logical'")
  }


  # Input formatting
  response_expr    <- dplyr::enquo(response)


  # Main function body

  # Determine truth
  truth_tbl <- new_data %>%
    dplyr::select(!!response_expr) %>%
    purrr::set_names("truth")

  if (.verbose)
    print(glue::glue("Truth column set to {quo_name(response_expr)}..."))

  # Calculate probabilities
  probs_tbl <- object %>%
    parsnip::predict.model_fit(new_data = new_data, type = "prob") %>%
    purrr::set_names(stringr::str_remove_all(names(.), ".pred_"))

  if (.verbose)
    print(glue::glue("Predicted probabilities calculated..."))

  # Determine positive outcome level
  if (is.null(positive)) {

    inferred_positive_level <- probs_tbl %>%
      tidyr::gather() %>%
      dplyr::group_by(key) %>%
      dplyr::summarize(mean_prob = mean(value)) %>%
      dplyr::ungroup() %>%
      dplyr::filter(mean_prob == max(mean_prob)) %>%
      dplyr::pull(key)

    if (.verbose)
      print(glue::glue("No positive outcome level given... inferred as {inferred_positive_level}"))

  } else {

    inferred_positive_level <- positive

  }

  # Calculate predicted class
  inferred_negative_level <- outcome_levels[outcome_levels != inferred_positive_level]

  preds_tbl <- probs_tbl %>%
    dplyr::select(inferred_positive_level) %>%
    purrr::set_names("Class1") %>%
    dplyr::mutate(
      prediction = dplyr::case_when(
        Class1 >= .thresh ~ inferred_positive_level,
        TRUE              ~ inferred_negative_level
      ) %>% factor(levels = outcome_levels)
    ) %>%
    dplyr::select(prediction)

  if (.verbose)
    print(glue::glue("Predicted response classes determined..."))


  output_tbl <- truth_tbl %>%
    dplyr::bind_cols(probs_tbl) %>%
    dplyr::bind_cols(preds_tbl)

  if (.verbose)
    print(glue::glue("Function completed successfully"))

  return(output_tbl)

}
