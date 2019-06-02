
## Motivation

The goal of this function is to produce a standardized output for
predictions using parsnip binary classification models.

The output of this function will include:

1.  The ‘truth’ (that is, what the actual outcome was)
2.  The predicted probability for each class (restricted to only two
    classes)
3.  The predicted response class based on the predicted probability of
    the “positive class”

<!-- end list -->

``` r
library(tidyverse)
library(glue)
library(tidymodels)
```

We will use the LendingClub data available in the parsnip package to
demonstrate how this function works.

``` r
class_data_tbl <- parsnip::lending_club %>%
  select(Class, funded_amnt:sub_grade, annual_inc, emp_length, revol_util)

class_data_tbl
```

    ## # A tibble: 9,857 x 8
    ##    Class funded_amnt term  int_rate sub_grade annual_inc emp_length
    ##    <fct>       <int> <fct>    <dbl> <fct>          <dbl> <fct>     
    ##  1 good        16100 term~    14.0  C4             35000 emp_5     
    ##  2 good        32000 term~    12.0  C1             72000 emp_ge_10 
    ##  3 good        10000 term~    16.3  D1             72000 emp_ge_10 
    ##  4 good        16800 term~    13.7  C3            101000 emp_lt_1  
    ##  5 good         3500 term~     7.39 A4             50100 emp_unk   
    ##  6 good        10000 term~    11.5  B5             32000 emp_lt_1  
    ##  7 good        11000 term~     5.32 A1             65000 emp_4     
    ##  8 good        15000 term~     9.16 B2            188000 emp_6     
    ##  9 good         6000 term~     9.8  B3             89000 emp_ge_10 
    ## 10 good        20000 term~    13.0  C2             48000 emp_ge_10 
    ## # ... with 9,847 more rows, and 1 more variable: revol_util <dbl>

``` r
split_obj <- initial_split(class_data_tbl, prop = 0.8, strata = "Class")
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)


model_logit_obj <- logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(Class ~ ., data = train_tbl)

tidy(model_logit_obj)
```

    ## # A tibble: 51 x 5
    ##    term           estimate    std.error statistic  p.value
    ##    <chr>             <dbl>        <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)  4.15         1.14          3.63   0.000280
    ##  2 funded_amnt -0.00000632   0.00000700   -0.904  0.366   
    ##  3 termterm_60  0.454        0.130         3.50   0.000460
    ##  4 int_rate     0.264        0.164         1.62   0.106   
    ##  5 sub_gradeA2 -0.422        1.24         -0.339  0.734   
    ##  6 sub_gradeA3 -1.72         0.957        -1.79   0.0729  
    ##  7 sub_gradeA4 11.5        249.            0.0460 0.963   
    ##  8 sub_gradeA5 -2.15         0.923        -2.33   0.0198  
    ##  9 sub_gradeB1 -2.46         0.944        -2.61   0.00913 
    ## 10 sub_gradeB2 -2.14         1.05         -2.03   0.0422  
    ## # ... with 41 more rows

## Manual Method

We can manually compute each component of this function’s planned output
as follows:

1.  The Truth

<!-- end list -->

``` r
truth_tbl <- test_tbl %>% 
  select(Class) %>% 
  set_names("truth")

truth_tbl
```

    ## # A tibble: 1,971 x 1
    ##    truth
    ##    <fct>
    ##  1 good 
    ##  2 good 
    ##  3 good 
    ##  4 good 
    ##  5 good 
    ##  6 good 
    ##  7 good 
    ##  8 good 
    ##  9 good 
    ## 10 good 
    ## # ... with 1,961 more rows

2.  The Predicted Probabilities

<!-- end list -->

``` r
probs_tbl <- model_logit_obj %>% 
  predict(new_data = test_tbl, type = "prob") %>% 
  set_names(str_remove_all(names(.), ".pred_"))

probs_tbl
```

    ## # A tibble: 1,971 x 2
    ##             bad  good
    ##           <dbl> <dbl>
    ##  1 0.0320       0.968
    ##  2 0.0000000230 1.000
    ##  3 0.0229       0.977
    ##  4 0.0195       0.981
    ##  5 0.0206       0.979
    ##  6 0.209        0.791
    ##  7 0.0907       0.909
    ##  8 0.0541       0.946
    ##  9 0.121        0.879
    ## 10 0.0249       0.975
    ## # ... with 1,961 more rows

3.  The Predicted Outcome Classes

<!-- end list -->

``` r
threshold <- 0.5
levels    <- test_tbl %>% pull(Class) %>% levels()

preds_tbl <- probs_tbl %>% 
  mutate(
    prediction = case_when(
      good >= threshold ~ "good",
      TRUE              ~ "bad"
    ) %>% factor(levels = levels)
  ) %>% 
  select(prediction)

preds_tbl
```

    ## # A tibble: 1,971 x 1
    ##    prediction
    ##    <fct>     
    ##  1 good      
    ##  2 good      
    ##  3 good      
    ##  4 good      
    ##  5 good      
    ##  6 good      
    ##  7 good      
    ##  8 good      
    ##  9 good      
    ## 10 good      
    ## # ... with 1,961 more rows

Can combine the previous together into the goal output tibble

``` r
output_tbl <- truth_tbl %>% 
  bind_cols(probs_tbl) %>% 
  bind_cols(preds_tbl)

output_tbl
```

    ## # A tibble: 1,971 x 4
    ##    truth          bad  good prediction
    ##    <fct>        <dbl> <dbl> <fct>     
    ##  1 good  0.0320       0.968 good      
    ##  2 good  0.0000000230 1.000 good      
    ##  3 good  0.0229       0.977 good      
    ##  4 good  0.0195       0.981 good      
    ##  5 good  0.0206       0.979 good      
    ##  6 good  0.209        0.791 good      
    ##  7 good  0.0907       0.909 good      
    ##  8 good  0.0541       0.946 good      
    ##  9 good  0.121        0.879 good      
    ## 10 good  0.0249       0.975 good      
    ## # ... with 1,961 more rows

## predict\_binary() Function

The predict\_binary() function automates the aforementioned process

``` r
model_predictions_tbl <- model_logit_obj %>% 
  predict_binary(new_data = test_tbl, response = Class, positive = "good")

model_predictions_tbl
```

    ## # A tibble: 1,971 x 4
    ##    truth          bad  good prediction
    ##    <fct>        <dbl> <dbl> <fct>     
    ##  1 good  0.0320       0.968 good      
    ##  2 good  0.0000000230 1.000 good      
    ##  3 good  0.0229       0.977 good      
    ##  4 good  0.0195       0.981 good      
    ##  5 good  0.0206       0.979 good      
    ##  6 good  0.209        0.791 good      
    ##  7 good  0.0907       0.909 good      
    ##  8 good  0.0541       0.946 good      
    ##  9 good  0.121        0.879 good      
    ## 10 good  0.0249       0.975 good      
    ## # ... with 1,961 more rows

The function provides various functions, such as .thresh that allows the
user to change the probability threshold that determines which the
outcome class is, and .verbose that keeps the user informed as to where
the function is executing

``` r
model_predictions_tbl <- model_logit_obj %>% 
  predict_binary(new_data = test_tbl, response = Class, positive = "good", .thresh = 0.95, .verbose = TRUE)
```

    ## Truth column set to Class...
    ## Predicted probabilities calculated...
    ## Predicted response classes determined...
    ## Function completed successfully

``` r
model_predictions_tbl
```

    ## # A tibble: 1,971 x 4
    ##    truth          bad  good prediction
    ##    <fct>        <dbl> <dbl> <fct>     
    ##  1 good  0.0320       0.968 good      
    ##  2 good  0.0000000230 1.000 good      
    ##  3 good  0.0229       0.977 good      
    ##  4 good  0.0195       0.981 good      
    ##  5 good  0.0206       0.979 good      
    ##  6 good  0.209        0.791 bad       
    ##  7 good  0.0907       0.909 bad       
    ##  8 good  0.0541       0.946 bad       
    ##  9 good  0.121        0.879 bad       
    ## 10 good  0.0249       0.975 good      
    ## # ... with 1,961 more rows
