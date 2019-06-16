tune\_cv\_reg()
================

## Motivation

The goal of this function is to automate k-fold cross-validation for
regression models within the tidymodels ecosystem

``` r
  library(MASS)
  library(tidyverse)
  library(tidymodels)
  library(caret)
  library(glue)
```

We will use the Boston Housing data to demonstrate training a random
forest regression model

``` r
  boston_tbl <- MASS::Boston %>%
    as_tibble() %>%
    rename(price = medv) %>%
    select(price, everything())
  
  glimpse(boston_tbl)
```

    ## Observations: 506
    ## Variables: 14
    ## $ price   <dbl> 24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, ...
    ## $ crim    <dbl> 0.00632, 0.02731, 0.02729, 0.03237, 0.06905, 0.02985, ...
    ## $ zn      <dbl> 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.5, 12.5, 12.5, 12.5,...
    ## $ indus   <dbl> 2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, ...
    ## $ chas    <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    ## $ nox     <dbl> 0.538, 0.469, 0.469, 0.458, 0.458, 0.458, 0.524, 0.524...
    ## $ rm      <dbl> 6.575, 6.421, 7.185, 6.998, 7.147, 6.430, 6.012, 6.172...
    ## $ age     <dbl> 65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 66.6, 96.1, 100.0,...
    ## $ dis     <dbl> 4.0900, 4.9671, 4.9671, 6.0622, 6.0622, 6.0622, 5.5605...
    ## $ rad     <int> 1, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, ...
    ## $ tax     <dbl> 296, 242, 242, 222, 222, 222, 311, 311, 311, 311, 311,...
    ## $ ptratio <dbl> 15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, ...
    ## $ black   <dbl> 396.90, 396.90, 392.83, 394.63, 396.90, 394.12, 395.60...
    ## $ lstat   <dbl> 4.98, 9.14, 4.03, 2.94, 5.33, 5.21, 12.43, 19.15, 29.9...

## Cross-validation in caret

Using the caret package, k-fold cross-validation was done using the
train() and trainControl() functions

``` r
  # set random seed
  set.seed(1234)
  
  # split into 80% training set and 20% testing set
  split_caret_obj <- createDataPartition(boston_tbl$price, p = .8, list = FALSE)
  train_caret_tbl <- boston_tbl[ split_caret_obj, ]
  test_caret_tbl  <- boston_tbl[-split_caret_obj, ]
  
  # 5-fold cross-validation
  ctrl <- trainControl(method = "cv", number = 5)
  
  # tuning grid for randomForest (mtry)
  tuning_grid_caret_obj <- expand.grid(mtry = c(4:7))
  
  # train the randomForest
  rf_fit_obj <- train(
    price ~ .,
    data       = train_caret_tbl,
    trControl  = ctrl,
    tuneGrid   = tuning_grid_caret_obj,
    ntree      = 100,
    importance = TRUE
  )
  
  rf_fit_obj
```

    ## Random Forest 
    ## 
    ## 407 samples
    ##  13 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 326, 326, 326, 325, 325 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared   MAE     
    ##   4     3.379306  0.8767422  2.268749
    ##   5     3.296260  0.8792113  2.237599
    ##   6     3.323944  0.8752525  2.292956
    ##   7     3.316003  0.8740218  2.242694
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 5.

## Cross-validation in parsnip & tidymodels

Using the tidymodels packages, k-fold cross-validation can be done using
tune\_cv\_reg()

``` r
  # set random seed
  set.seed(1234)

  # split into 80% training set and 20% testing set
  split_rsample_obj <- rsample::initial_split(boston_tbl, prop = 0.8)
  train_rsample_tbl <- rsample::training(split_rsample_obj)
  test_rsample_tbl  <- rsample::testing(split_rsample_obj)
  
  # set up the base random forest object
  model_base_obj <- parsnip::rand_forest(mode = "regression") %>%
    parsnip::set_engine("randomForest") %>%
    parsnip::set_args(mtry = varying(), trees = 100, importance = TRUE)
    
  # tuning grid for randomForest (mtry)
  tuning_grid_dials_obj <- dials::grid_regular(
    dials::mtry  %>% range_set(c(4, 7)),
    levels = 4
  )
  
  # cross-validate the random forest
  tuning_grid_metrics_tbl <- model_base_obj %>% 
    tune_cv_reg(
      data     = train_rsample_tbl,
      grid     = tuning_grid_dials_obj,
      response = price,
      k        = 5,
      n        = 1
    )
  
  tuning_grid_metrics_tbl
```

    ## # A tibble: 4 x 4
    ##    mtry  rmse   rsq   mae
    ##   <int> <dbl> <dbl> <dbl>
    ## 1     4  3.42 0.866  2.29
    ## 2     5  3.41 0.864  2.25
    ## 3     6  3.35 0.869  2.21
    ## 4     7  3.42 0.860  2.22

``` r
  tuning_grid_metrics_tbl %>% 
    arrange(rmse)
```

    ## # A tibble: 4 x 4
    ##    mtry  rmse   rsq   mae
    ##   <int> <dbl> <dbl> <dbl>
    ## 1     6  3.35 0.869  2.21
    ## 2     5  3.41 0.864  2.25
    ## 3     7  3.42 0.860  2.22
    ## 4     4  3.42 0.866  2.29

## Current status and future updates

Note that this function is very limited, and is nowhere near
“production-ready”. I developed it to help tune parameters for a few
regression models I was building for work, and thought it might be a
useful bit of code for anyone who may run across it.

Future iterations will include full beginning-to-end automatic
hyperparameter tuning and training. The output will be a list that
includes items such as information about the base model, the training
data supplied to the function, the rsample vfold\_cv() result that is
used inside the function, the final model trained with the “best”
parameters, etc.

I hope to devote more time to building out its functionality in the
future, but it is more likely that the R Studio team (or someone much
better at R than I am) creates their own automatic parameter tuning
functions for tidymodels way before then.
