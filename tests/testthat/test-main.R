# tests/testthat/test-temper.R
library(testthat)
library(unfold)


# tests/testthat/helper-torch.R
has_working_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) return(FALSE)
  tryCatch({
    torch::torch_tensor(1)$item()
    TRUE
  }, error = function(e) FALSE)
}

skip_if_no_working_torch <- function() {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not(
    has_working_torch(),
    "Skipping because torch runtime/Lantern is unavailable."
  )
}

set.seed(42)

ts_set <- data.frame(
  A = cumsum(rnorm(100)),
  B = cumsum(rnorm(100))
)

test_that("unfold returns expected object structure", {
  #skip_if_not_installed("torch")
  skip_if_no_working_torch()

  fit <- unfold(ts_set, horizon = 2, epochs = 2, batch_size = 8, verbose = FALSE)

  expect_type(fit, "list")
  expect_named(fit,
               c("description", "model", "dist_array", "loss_plot",
                 "pred_funs", "graph_plot", "time_log"),
               ignore.order = TRUE
  )
  expect_s3_class(fit$model, "vam_fit")
  expect_true(is.array(fit$dist_array))
})

test_that("predictive functions exist and are callable", {
  #skip_if_not_installed("torch")
  skip_if_no_working_torch()

  fit <- unfold(ts_set, horizon = 2, epochs = 2, batch_size = 8, verbose = FALSE)

  expect_true("A" %in% names(fit$pred_funs))
  pfuns <- fit$pred_funs$A
  expect_equal(names(pfuns), c("t1","t2"))

  # gmix() closures: at least they should be functions
  expect_true(all(vapply(pfuns$t2, is.function, logical(1))))
})

test_that("unfold is reproducible with fixed seed (within tolerance)", {
  #skip_if_not_installed("torch")
  skip_if_no_working_torch()

  fit1 <- unfold(ts_set[, 1, drop = FALSE], horizon = 2, seed = 123, epochs = 2, batch_size = 8, verbose = FALSE)
  fit2 <- unfold(ts_set[, 1, drop = FALSE], horizon = 2, seed = 123, epochs = 2,  batch_size = 8, verbose = FALSE)

  # Compare numeric histories within tolerance
  expect_equal(
    fit1$model$history,
    fit2$model$history,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
})


