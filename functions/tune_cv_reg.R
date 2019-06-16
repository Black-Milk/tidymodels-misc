# Automated hyperparmeter tuning of parsnip tidymodels using k-fold cross-validation

# Utility Functions ----

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
    estimate = parsnip::predict.model_fit(object, new_data = new_data)$.pred
  )

}


run_parameters <- function(object, iteration, cv, response_expr, .verbose) {

  model_formula <- as.formula(glue::glue("{dplyr::quo_name(response_expr)} ~ ."))

  fit_model_partial       <- purrr::partial(fit_model, object = object, formula = model_formula, .verbose = .verbose)
  predict_reg_partial     <- purrr::partial(predict_reg, response_expr = response_expr)
  regress_metrics_partial <- purrr::partial(regress_metrics, truth = truth, estimate = estimate)

  output_tbl <- cv %>%
    dplyr::mutate(
      analysis_tbl      = purrr::map(splits, analysis),
      assessment_tbl    = purrr::map(splits, assessment),
      model_trained_obj = purrr::pmap(list(data = analysis_tbl, iter = iteration, run = id, fold = id2), fit_model_partial),
      predictions_tbl   = purrr::map2(model_trained_obj, assessment_tbl, predict_reg_partial),
      metrics_tbl       = purrr::map(predictions_tbl, regress_metrics_partial)
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


# Main Function ----

tune_cv_reg <- function(object, data, grid, response,
                        k = 5, n = 3, .verbose = FALSE) {

  # Inputs:
  #   object   : an 'un-trained' parsnip model object (that is, one that has
  #              varying() for at least one parameter, and has not been piped
  #              into fit())
  #   data     : a data frame or tibble of training data, where the response
  #              variable (see later) will be regressed against all other
  #              variables in the data set
  #   grid     : a grid_random or grid_regular tibble made with the dials
  #              package that contains unique possible values for all
  #              parameters in object that are varying()
  #   response : the unquoted column name of the response variable
  #              (must be continuous for regression)
  #   k        : the number of folds (must be a positive integer)
  #   n        : the number of repeats (must be a positive integer)
  #   .verbose : if TRUE, messages will be printed indicating when each
  #              combination of parameter/repeat/fold has been fit to a
  #              model (to keep track of the progress)
  #
  #
  # Output:
  #   The original grid tibble appended with the average value of the
  #   calculated performance metrics from all repeats/folds that had models
  #   trained with those parameter values
  #
  #
  # Outline:
  #   1. 'Input Validation' makes sure input does not violate any restrictions
  #      as described in above in the Inputs section
  #   2. 'Input Formatting' makes sure input matches the format described above
  #      in the Inputs section
  #   3. 'Partialize Functions' fills in the arguments to the utility functions
  #      that do not change through any iteration
  #   4. 'Run Functions' runs all tidymodels functions and utility functions
  #       as follows (Note: steps 4.1-4.3 are in the 'Input Formatting' step):
  #       4.1. Merges the base object (un-trained parsnip model) with the
  #            hyperparameter grid to make a list of model ready to be
  #            piped into fit()
  #       4.2. Create the formula for the model (response regressed
  #            against all other variables)
  #       4.3. Create a cross-validation (cv) object using rsample::vfold_cv()
  #       4.4. For each model in the list made in 4.1
  #            4.5. Take the Cv object and map rsample::analysis() and
  #                 rsample::assessment() to each repeat/fold combination
  #                 to get the analysis dataset and the assessment dataset
  #            4.6. Fit a model with the corresponding parameter combinations
  #                 using the analysis dataset (print a message if
  #                 .verbose is set to TRUE)
  #            4.7. Calculate performance metrics (RMSE, r-squared, and MAE by
  #                 default) using the assessment dataset by calculating the
  #                 predicted outcome values of the assessment dataset from
  #                 each model, and piping those into the corresponding
  #                 yardstick package functions (e.g. rmse(), rsq(), and mae())
  #            4.8. Average performance metrics across each repeat/fold for
  #                 each model to get a final performance metric of each type
  #                 for each parameter combination
  #            4.9  Combine the metrics together with the original parameter
  #                 grid and output


  # Input Validation
  if (!inherits(object, "model_spec")) {
    stop("Argument 'object' must be a parsnip package model object")
  } else if (!inherits(data, c("data.frame", "tbl"))) {
    stop("Argument 'data' must be a data frame or tibble")
  } else if (!inherits(grid, "param_grid")) {
    stop("Argument 'grid' must be a dials package grid_random or grid_regular")
  } else if (!(n_distinct(grid) == nrow(grid))) {
    stop("Argument 'grid' must have all unique rows (parameter combinations must be unique)")
  } else if (!(quo_name(enquo(response)) %in% names(data))) {
    stop("Argument 'response' must be an unquoted valid column name from the 'data' argument")
  } else if (!inherits(dplyr::pull(data, !!enquo(response)), "numeric")) {
    stop("This function is for regressions, argument 'response' must reference a variable of type 'numeric'")
  } else if (!(inherits(as.integer(k), "integer") & inherits(as.integer(n), "integer") & as.integer(k) > 0 & as.integer(n) > 0)) {
    stop("Arguments 'k' and 'n' must be of type integer and both be greater than zero")
  } else if (!inherits(.verbose, "logical")) {
    stop("Argument '.verbose' must be of type 'logical'")
  }


  # Input formatting
  response_expr    <- dplyr::enquo(response)
  num_param_combos <- nrow(grid)
  iteration_count  <- seq(from = 1, to = num_param_combos, by = 1)
  model_obj_list   <- merge(object, grid)
  model_formula    <- as.formula(glue::glue("{dplyr::quo_name(response_expr)} ~ ."))
  k_int            <- as.integer(k)
  n_int            <- as.integer(n)
  cv_obj           <- rsample::vfold_cv(data, v = k_int, repeats = n_int)

  if (n == 1)
    cv_obj <- cv_obj %>% dplyr::rename(id2 = id) %>% dplyr::mutate(id = "Repeat1")


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

