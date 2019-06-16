# Automated aggregation of necessary prediction results for classification models

# Main Function ----

predict_binary <- function(object, new_data, response, positive = NULL,
                           .thresh = 0.5, .verbose = FALSE) {

  # Inputs:
  #   Object   : a 'fully trained' parsnip model object ready to be piped into predict()
  #   new_data : a data frame or tibble exactly matching the training data on which the
  #              object was trained (typically, the testing data set)
  #   response : the unquoted column name of the response variable
  #              (must be a two-level factor for classification)
  #   positive : an optional character string indicating the 'positive' outcome level of the response column
  #              (if null, function attempts to determine it by taking the level with the higher average
  #               predicted probability)
  #   .thresh  : an optional numeric indicating at what probability we classify everything above the value
  #              as the 'positive' outcome, and everything below it as the 'negative' outcome
  #   .verbose : if TRUE, messages will be printed indicating when each
  #              combination of parameter/repeat/fold has been fit to a
  #              model (to keep track of the progress)
  #
  #
  # Output:
  #   A tibble with the following four columns:
  #     truth      : the actual outcome classes from new_data
  #     Class1     : the predicted probability of the 'positive class' (will have the column name changed)
  #     Class2     : the predicted probability of the 'negative class' (will have the column name changed)
  #     prediction : the predicted outcome class based on arguments 'positive' and '.thresh'
  #
  #
  # Outline:
  #   1. 'Input Validation' makes sure input does not violate any restrictions
  #      as described in above in the Inputs section
  #   2. 'Input Formatting' makes sure input matches the format described above
  #      in the Inputs section
  #   3. 'Main Function' works as follows:
  #       3.1. Pull the 'truth' column (the response) from new_data
  #       3.2. Calculate the predicted probabilities of each outcome class
  #            (each of Class1 and Class2 will have the column name changed to the actual class name)
  #            (if 'positive' is set to NULL, the level will be inferred by assuming that the
  #             with the highest average probability is the 'positive' class)
  #       3.3. Calculate the predicted outcome classes based on 'positive' and '.thresh' arguments
  #            (if the 'positive' class probability is greater than or equal to .thresh,
  #             assign it to the 'positive' class; if less than, assign to 'negative')
  #       3.4. Combine columns of truth, each class probabilities, and predicted class into a single
  #            outcome tibble and output


  # Input Validation
  if (!inherits(object, "model_fit")) {
    stop("Argument 'object' must be a parsnip package model object")
  } else if (object$spec$mode != "classification") {
    stop("Argument 'object' must be a classification model")
  } else if (!inherits(new_data, c("data.frame", "tbl"))) {
    stop("Argument 'new_data' must be a data frame or tibble")
  } else if (!(quo_name(enquo(response)) %in% names(new_data))) {
    stop("Argument 'response' must be an unquoted valid column name from the 'new_data' argument")
  } else if (!inherits(dplyr::pull(new_data, !!enquo(response)), "factor")) {
    stop("This function is for classifications, argument 'response' must reference a variable of type 'factor'")
  } else if (length(levels(dplyr::pull(new_data, !!enquo(response)))) != 2) {
    stop("This function is for binary classifications, argument 'response' must have only two levels")
  } else if (!is.null(positive)) {
    if (!(positive %in% levels(dplyr::pull(new_data, !!enquo(response))))) {
      stop("If non-NULL, argument 'positive' must match one of the factor levels in the column referenced by argument 'response'")
    }
  } else if (!dplyr::between(.thresh, 0, 1)) {
    stop("Argument '.thresh' must be of type 'numeric' and must be between 0 and 1")
  } else if (!inherits(.verbose, "logical")) {
    stop("Argument '.verbose' must be of type 'logical'")
  }


  # Input formatting
  response_expr  <- dplyr::enquo(response)
  outcome_levels <- new_data %>% pull(!!response_expr) %>% levels()


  # Main Function

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
