
## Miscellaneous tidymodels functions

This repository is a place for me to develop and test functions that
build upon the tidymodels ecosystem. I develop new functions as they are
needed in my day-to-day work.

**Current Functions**:

  - tune\_cv\_reg() : automated k-fold cross-validation hyperparameter
    tuning for regression-based parsnip models
  - predict\_binary() : standardize output of predictions on
    classification-based parsnip models (truth, class probabilities, and
    outcome class)

**In Development**:

  - tune\_cv\_class() : automated k-fold cross-validation hyperparameter
    tuning for classification-based parsnip models

**Future Ideas**:

  - combine tune\_cv\_reg() and tune\_cv\_class() into a single
    tune\_cv() function
  - create parallelized variants of tune\_cv\_reg() and
    tune\_cv\_class() using furrr package
