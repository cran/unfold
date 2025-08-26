#' unfold: Mapping Hidden Geometry into Future Sequences
#'
#' @param ts_set A data frame containing the time series, one column per series.
#' @param horizon Integer. Forecast horizon; controls reframing and output functions.
#' @param metric Distance metric fro the 4D tensor; one of "euclidean", "mahalanobis", "cosine". Default: "euclidean".
#' @param latent_dim Integer. Latent dimensionality of the variational mapper. Default: 32.
#' @param enc_hidden,dec_hidden Integer vectors. Hidden layer widths for encoder/decoder MLPs, defaulting to c(512, 256) and c(256, 512) respectively.
#' @param p_drop Dropout probability in encoder/decoder. Default: 0.1.
#' @param out_kind Output nonlinearity of the decoder; one of "linear", "tanh" (used by the VAM). Default: "linear".
#' @param epochs Integer. Training epochs. Default: 30.
#' @param batch_size Integer. Dimension of batch. Default: 64.
#' @param lr Double. Learning rate. Default: 1e-3.
#' @param beta Double. KL weight for the variational objective. Default: 1.
#' @param beta_warmup Integer. If > 0, linearly warm up beta over this many epochs. Default: 0.
#' @param grad_clip Optional max norm for gradient clipping. If you never see exploding losses or NaNs, you can leave it NULL, otherwise, if training diverges, try clipping (1 to 5) and monitor if loss becomes smoother. Default: NULL.
#' @param valid_split Double. Proportion of samples held out for validation during VAM training. Default: 0.2.
#' @param verbose Logical. Print training progress. Default: TRUE.
#' @param alpha Double. Forecasting confidence interval used in plotted graphs. Default: 0.1.
#' @param dates Character. Vector with the original time series dates in text format, used for plotting purposes. Default: NULL.
#' @param patience Integer Epochs of stagnation before early stopping. Default: NULL.
#' @param n_bases Integer Maximum number of distributions to use for the Gaussian mixture. Default: 10.
#' @param seed Random seed for reproducibility. Default: 42.
#'
#' @return A named list with the following components:
#' \describe{
#'   \item{`description`}{Character string giving a short description of the model (parameters, activations and so on).}
#'   \item{`model`}{A fitted variational mapper object of class vam_fit. This object contains the trained network plus helper methods (encode, decode, reconstruct, predict, etc.).}
#'   \item{`dist_array`}{A numeric 4D array containing pairwise distances between reframed time-series windows: shape N x N x M x M, where N is the number of reframed time-series windows and M the number of time series.}
#'   \item{`loss_plot`}{A ggplot plot object showing the training and validation loss curves across epochs.}
#'   \item{`pred_funs`}{For each time series, a length-horizon list containing four gaussian mix distribution functions (dfun, pfun, qfun, rfun).}
#'   \item{`graph_plot`}{A list including ggplot graphs for each time series reproducing the predicted horizon with confidence interval alpha.}
#'   \item{`time_log`}{An object measuring the elapsed time for the computation (preprocessing, training, prediction, etc.).}
#' }
#'
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE)) {
#'   set.seed(42)
#'
#'   # --- Create a small synthetic dataset with 3 series ---
#'   T <- 100
#'   ts_set <- data.frame(
#'     A = cumsum(rnorm(T, mean = 0.02, sd = 0.1)) + 10,
#'     B = cumsum(rnorm(T, mean = 0.01, sd = 0.08)) +  8,
#'     C = cumsum(rnorm(T, mean = 0.00, sd = 0.12)) + 12
#'   )
#'
#'   # --- Fit the model ---
#'   fit <- unfold(
#'     ts_set    = ts_set,
#'     horizon   = 3,
#'     metric    = "euclidean",
#'     latent_dim  = 16,
#'     enc_hidden  = c(64, 32),
#'     dec_hidden  = c(32, 64),
#'     epochs      = 5,
#'     batch_size  = 16,
#'     verbose     = FALSE
#'   )
#'
#'   # --- Inspect predictive functions ---
#'   names(fit$pred_funs)         # series names
#'   names(fit$pred_funs$A)       # "t1" "t2" "t3"
#'
#'   # Example: call predictive function for series A, horizon t1
#'   f_t1 <- fit$pred_funs$A$t1$rfun
#'   # Example: draw 500 simulated values
#'   # sims <- f_t1(500)
#' }
#' }
#'
#' @import torch ggplot2 purrr
#' @importFrom stats  quantile runif dunif punif qunif bw.nrd0 bw.nrd dnorm approxfun fft median sd rt coef pnorm rnorm splinefun cov kmeans
#' @importFrom imputeTS  na_kalman
#' @importFrom scales  number
#' @importFrom lubridate  seconds_to_period period
#' @importFrom utils  tail head
#' @importFrom coro loop
#' @importFrom abind abind
#'
#'
#' @export
unfold <- function(ts_set, horizon, metric = "euclidean",
                   latent_dim = 32, enc_hidden = c(512, 256), dec_hidden = c(256, 512),
                   p_drop = 0.1, out_kind = "linear", epochs = 30,
                   batch_size = 64, lr = 1e-3, beta = 1.0, beta_warmup = 0,
                   grad_clip = NULL, valid_split = 0.2,
                   verbose = TRUE, alpha = 0.1, dates = NULL, patience = NULL, n_bases = 10, seed = 42) {

  start <- Sys.time()
  max_horizon <- floor((nrow(ts_set)-2)/(ceiling(30/(1 - valid_split))))

  stopifnot(is.data.frame(ts_set), horizon > 0)
  if(horizon > max_horizon){horizon <- max_horizon; message("horizon adapted to available data\n")}

  set.seed(seed)

  out_kind <- match.arg(out_kind, c("linear", "tanh"), several.ok=FALSE)
  metric <- match.arg(metric, c("euclidean", "mahalanobis", "cosine"), several.ok=FALSE)

  n_ts <- ncol(ts_set)
  feat_names <- colnames(ts_set)

  reframed <- purrr::map(ts_set, ~ as.data.frame(smart_reframer(dts(.x, 1), horizon, horizon)))
  reframed_array <- abind::abind(reframed, along = 3)
  dist_array <- distances4d(reframed_array, metric, cov = NULL, method = "auto",
                            symmetric_blocks = TRUE, max_mat_elems = 4e7, zero_upper = "both", zero_diag_within = TRUE)

  reframed_array <- torch_tensor(reframed_array)
  dist_array <- torch_tensor(dist_array)

  model <- train_variational_mapper(X = head(dist_array, -1), Y = tail(reframed_array, -1), mask = NULL, latent_dim, enc_hidden, dec_hidden,
                                    p_drop, out_kind, target_shape = NULL, epochs, batch_size, lr, loss_kind = "mse", beta, beta_warmup,
                                    grad_clip, valid_split, shuffle = TRUE, device = NULL, verbose, patience, seed)

  new <- tail(dist_array, 1)

  preds <- model$predict(new, n_samples = 1000)$recon
  preds <- preds$reshape(c(1000, horizon, n_ts))
  preds <- as.array(preds)

  last_level <- as.numeric(tail(ts_set,1))

  proj_space <- lapply(1:n_ts, function(ts) t(apply(preds[,,ts,drop = FALSE], 1, function(r) last_level[ts] * cumprod(1 + r))))

  if(horizon == 1){proj_space <- lapply(proj_space, function(ts) t(ts))}

  pred_funs <- lapply(1:n_ts, function(ts) map(data.frame(proj_space[[ts]]), ~ gmix(.x, n_bases, seed)))
  pred_funs <- lapply(pred_funs, function(pf) {names(pf) <- paste0("t", 1:horizon); return(pf)})
  names(pred_funs) <- feat_names

  graph_plot <- map(feat_names, ~plot_graph(ts_set[[.x]], pred_funs[[.x]], alpha = alpha, dates = dates))
  names(graph_plot) <- feat_names

  loss_plot <- ggplot(model$history, aes(.data$epoch)) +
    geom_line(aes(y = .data$train_loss, colour = "train"), alpha = 0.8) +
    geom_line(aes(y = .data$valid_loss,   colour = "val"),   alpha = 0.8) +
    scale_colour_manual(values = c("train" = "#1b9e77", "val" = "#d95f02")) +
    theme_minimal() + labs(y = "loss", colour = "split")

  description <- model$desc

  end <- Sys.time()
  time_log <- seconds_to_period(round(difftime(end, start, units = "secs"), 0))

  out <- list(description = description, model = model, dist_array = as.array(dist_array), loss_plot = loss_plot, pred_funs = pred_funs, graph_plot = graph_plot, time_log = time_log)

  return(out)
}


# ---------------------------------------------------------------------------
# Everything below is INTERNAL
# ---------------------------------------------------------------------------

globalVariables(c(".."))

#' @keywords internal
distances4d <- function(
    X,
    metric = c("euclidean","mahalanobis","cosine"),
    cov = NULL,
    method = c("auto","global","block"),
    symmetric_blocks = TRUE,
    max_mat_elems = 4e7,        # ~320 MB for doubles
    zero_upper = c("none","blocks","within","both"),
    zero_diag_within = FALSE    # if TRUE, also zero block diagonals when zeroing within
) {
  stopifnot(is.array(X), length(dim(X)) == 3L)
  metric <- match.arg(metric)
  method <- match.arg(method)
  zero_upper <- match.arg(zero_upper)

  n <- dim(X)[1]; s <- dim(X)[2]; m <- dim(X)[3]
  if (n < 1 || s < 1 || m < 1) stop("Empty dimension in X")

  # ----- kernels -----
  euclid_cross <- function(A, B) {
    AA <- rowSums(A*A); BB <- rowSums(B*B); C <- A %*% t(B)
    D2 <- outer(AA, BB, "+") - 2*C
    sqrt(pmax(D2, 0))
  }
  cosine_cross <- function(A, B) {
    nA <- sqrt(rowSums(A*A)); nA[nA == 0] <- 1
    nB <- sqrt(rowSums(B*B)); nB[nB == 0] <- 1
    A1 <- A / nA; B1 <- B / nB
    S  <- A1 %*% t(B1)
    pmax(0, 1 - S)
  }
  build_mahal <- function() {
    if (is.null(cov)) {
      Y <- matrix(aperm(X, c(1,3,2)), ncol = s)
      S <- cov(Y)
    } else {
      S <- cov
      if (!is.matrix(S) || any(dim(S) != s)) stop("cov must be sxs")
    }
    tryCatch(solve(S), error = function(e) solve(S + diag(1e-8, s)))
  }
  mahal_cross_factory <- function(Sinv) {
    function(A, B) {
      AS <- A %*% Sinv; BS <- B %*% Sinv
      qA <- rowSums(AS * A); qB <- rowSums(BS * B)
      cross <- AS %*% t(B)
      D2 <- outer(qA, qB, "+") - 2 * cross
      sqrt(pmax(D2, 0))
    }
  }

  Sinv <- NULL
  xdist_cross <- switch(metric,
                        euclidean   = euclid_cross,
                        cosine      = cosine_cross,
                        mahalanobis = { Sinv <- build_mahal(); mahal_cross_factory(Sinv) }
  )

  D <- array(0, dim = c(n, n, m, m))

  # ----- compute (same as before) -----
  R <- n * m
  if (method == "auto") {
    method <- if (as.double(R) * as.double(R) <= max_mat_elems) "global" else "block"
  }

  if (method == "global") {
    Y <- matrix(aperm(X, c(1,3,2)), ncol = s)
    idx_block <- function(i) ((i-1)*n + 1):(i*n)

    if (metric == "euclidean") {
      norms <- rowSums(Y*Y)
      G <- Y %*% t(Y)
      for (i in 1:m) {
        ii <- idx_block(i)
        jseq <- if (symmetric_blocks) i:m else 1:m
        for (j in jseq) {
          jj <- idx_block(j)
          D2 <- outer(norms[ii], norms[jj], "+") - 2*G[ii, jj]
          Mij <- sqrt(pmax(D2, 0))
          D[,,i,j] <- Mij
          if (symmetric_blocks && j != i) D[,,j,i] <- t(Mij)
        }
      }
    } else if (metric == "cosine") {
      nY <- sqrt(rowSums(Y*Y)); nY[nY == 0] <- 1
      Y1 <- Y / nY
      Sg <- Y1 %*% t(Y1)
      for (i in 1:m) {
        ii <- idx_block(i)
        jseq <- if (symmetric_blocks) i:m else 1:m
        for (j in jseq) {
          jj <- idx_block(j)
          Mij <- pmax(0, 1 - Sg[ii, jj])
          D[,,i,j] <- Mij
          if (symmetric_blocks && j != i) D[,,j,i] <- t(Mij)
        }
      }
    } else { # mahalanobis
      Yt <- Y %*% Sinv
      q  <- rowSums(Yt * Y)
      Gm <- Yt %*% t(Y)
      for (i in 1:m) {
        ii <- idx_block(i)
        jseq <- if (symmetric_blocks) i:m else 1:m
        for (j in jseq) {
          jj <- idx_block(j)
          D2 <- outer(q[ii], q[jj], "+") - 2 * Gm[ii, jj]
          Mij <- sqrt(pmax(D2, 0))
          D[,,i,j] <- Mij
          if (symmetric_blocks && j != i) D[,,j,i] <- t(Mij)
        }
      }
    }

  } else { # block
    if (metric == "euclidean") {
      Ai <- lapply(1:m, function(i) X[,,i, drop=FALSE][,,1])
      norms <- lapply(Ai, function(A) rowSums(A*A))
      for (i in 1:m) {
        jseq <- if (symmetric_blocks) i:m else 1:m
        for (j in jseq) {
          C <- Ai[[i]] %*% t(Ai[[j]])
          D2 <- outer(norms[[i]], norms[[j]], "+") - 2*C
          Mij <- sqrt(pmax(D2, 0))
          D[,,i,j] <- Mij
          if (symmetric_blocks && j != i) D[,,j,i] <- t(Mij)
        }
      }
    } else if (metric == "cosine") {
      Ai <- vector("list", m)
      for (i in 1:m) {
        A <- X[,,i, drop=FALSE][,,1]
        nA <- sqrt(rowSums(A*A)); nA[nA == 0] <- 1
        Ai[[i]] <- A / nA
      }
      for (i in 1:m) {
        jseq <- if (symmetric_blocks) i:m else 1:m
        for (j in jseq) {
          S <- Ai[[i]] %*% t(Ai[[j]])
          Mij <- pmax(0, 1 - S)
          D[,,i,j] <- Mij
          if (symmetric_blocks && j != i) D[,,j,i] <- t(Mij)
        }
      }
    } else { # mahalanobis
      Sinv <- build_mahal()
      A  <- lapply(1:m, function(i) X[,,i, drop=FALSE][,,1])
      AS <- lapply(A, function(M) M %*% Sinv)
      q  <- lapply(1:m, function(i) rowSums(AS[[i]] * A[[i]]))
      for (i in 1:m) {
        jseq <- if (symmetric_blocks) i:m else 1:m
        for (j in jseq) {
          cross <- AS[[i]] %*% t(A[[j]])
          D2 <- outer(q[[i]], q[[j]], "+") - 2 * cross
          Mij <- sqrt(pmax(D2, 0))
          D[,,i,j] <- Mij
          if (symmetric_blocks && j != i) D[,,j,i] <- t(Mij)
        }
      }
    }
  }

  # ----- zeroing redundant symmetries (post-process, O(m^2) light) -----
  if (zero_upper %in% c("blocks","both")) {
    # zero the upper triangle in block space: D[,,i,j] <- 0 for j > i
    if (m > 1) {
      for (i in 1:(m-1)) {
        if (i+1 <= m) D[,,i,(i+1):m] <- 0
      }
    }
  }

  if (zero_upper %in% c("within","both")) {
    # zero upper triangle within each nxn block
    U <- upper.tri(matrix(0, n, n), diag = zero_diag_within)
    for (i in 1:m) {
      for (j in 1:m) {
        Bij <- D[,,i,j]
        Bij[U] <- 0
        D[,,i,j] <- Bij
      }
    }
  }

  D
}


#' @keywords internal
masked_reconstruction_loss <- function(recon, target, mask = NULL,
                                       kind = c("mse", "bce_logits")) {
  kind <- match.arg(kind)
  if (!is.null(mask)) {
    if (kind == "mse") {
      err <- (recon - target)$pow(2) * mask
      msum <- torch_sum(mask)
      return(torch_sum(err) / (msum + 1e-8))
    } else {
      x <- recon; y <- target
      max0 <- torch_clamp(x, min = 0)
      bce  <- max0 - x * y + torch_log1p(torch_exp(-torch_abs(x)))
      bce  <- bce * mask
      msum <- torch_sum(mask)
      return(torch_sum(bce) / (msum + 1e-8))
    }
  } else {
    if (kind == "mse") {
      return(torch_mean((recon - target)$pow(2)))
    } else {
      return(nnf_binary_cross_entropy_with_logits(
        recon, target, reduction = "mean"
      ))
    }
  }
}


#' @keywords internal
kl_standard_normal <- function(mu, logvar) {
  torch_mean(-0.5 * torch_sum(1 + logvar - mu$pow(2) - torch_exp(logvar), dim = 2))
}


#' @keywords internal
variational_auto_mapper <- nn_module(
  classname = "variational_auto_mapper",

  initialize = function(latent_dim   = 32,
                        enc_hidden   = c(512, 256),
                        dec_hidden   = c(256, 512),
                        p_drop       = 0.1,
                        nonlinearity = nn_silu,
                        out_kind     = c("linear", "tanh", "sigmoid", "logits"),
                        target_shape = NULL) {

    self$latent_dim   <- latent_dim
    self$enc_hidden   <- enc_hidden
    self$dec_hidden   <- dec_hidden
    self$p_drop       <- p_drop
    self$act          <- nonlinearity()
    self$flatten      <- nn_flatten(start_dim = 2)
    self$out_kind     <- match.arg(out_kind)
    self$user_tshape  <- target_shape

    # Lazy-built vars
    self$enc <- NULL
    self$fc_mu <- NULL
    self$fc_lv <- NULL
    self$dec <- NULL
    self$fc_out <- NULL

    self$feat_in  <- NULL
    self$feat_out <- NULL
    self$built    <- FALSE
  },

  .build_encoder = function(Fin) {
    dims <- c(Fin, self$enc_hidden)
    layers <- list()
    for (i in seq_len(length(dims) - 1)) {
      layers <- append(layers, list(
        nn_linear(dims[i], dims[i + 1]),
        self$act,
        nn_dropout(p = self$p_drop)
      ))
    }
    self$enc <- do.call(nn_sequential, layers)
    last_h <- tail(self$enc_hidden, 1)
    self$fc_mu <- nn_linear(last_h, self$latent_dim)
    self$fc_lv <- nn_linear(last_h, self$latent_dim)
  },

  .build_decoder = function(Fout) {
    dims <- c(self$latent_dim, self$dec_hidden)
    layers <- list()
    for (i in seq_len(length(dims) - 1)) {
      layers <- append(layers, list(
        nn_linear(dims[i], dims[i + 1]),
        self$act,
        nn_dropout(p = self$p_drop)
      ))
    }
    self$dec <- do.call(nn_sequential, layers)
    last_h <- tail(self$dec_hidden, 1)
    self$fc_out <- nn_linear(last_h, Fout)
  },

  .ensure_built = function(x) {
    if (self$built) return(invisible(NULL))

    x_flat <- self$flatten(x)
    self$feat_in <- as.integer(x_flat$size(2))
    self$.build_encoder(self$feat_in)

    if (is.null(self$user_tshape)) {
      tshape <- as.integer(x$size())[-1]   # drop batch dim
    } else {
      tshape <- as.integer(self$user_tshape)
    }
    self$target_shape <- tshape
    self$feat_out <- as.integer(prod(tshape))
    self$.build_decoder(self$feat_out)

    self$built <- TRUE
    invisible(NULL)
  },

  encode = function(x, sample = TRUE) {
    x_flat <- self$flatten(x)
    h  <- self$enc(x_flat)
    mu <- self$fc_mu(h)
    lv <- self$fc_lv(h)
    if (sample) {
      std <- torch_exp(0.5 * lv)
      eps <- torch_randn_like(std)
      z   <- mu + eps * std
    } else {
      z <- mu
    }
    list(z = z, mu = mu, logvar = lv)
  },

  decode = function(z) {
    h  <- self$dec(z)
    y  <- self$fc_out(h)
    if (self$out_kind == "tanh") {
      y <- torch_tanh(y)
    } else if (self$out_kind == "sigmoid") {
      y <- torch_sigmoid(y)
    } else if (self$out_kind == "logits") {
      y <- y
    } else {
      y <- y
    }
    B <- as.integer(y$size(1))
    y$view(c(B, self$target_shape))
  },

  forward = function(x, sample = TRUE) {
    self$.ensure_built(x)
    enc <- self$encode(x, sample = sample)
    recon <- self$decode(enc$z)
    list(recon = recon, z = enc$z, mu = enc$mu, logvar = enc$logvar)
  }
)


#' @keywords internal
vam_dataset <- dataset(
  name = "vam_dataset",
  initialize = function(X, Y = NULL, mask = NULL) {
    self$X    <- X
    self$Y    <- Y
    self$mask <- mask
    self$N    <- as.integer(X$size(1))
  },
  .getitem = function(i) {
    x <- self$X[i,..]
    y <- if (!is.null(self$Y)) self$Y[i,..] else x
    if (!is.null(self$mask)) list(x = x, y = y, mask = self$mask[i,..]) else list(x = x, y = y)
  },
  .length = function() self$N
)


#' @keywords internal
fit_variational_auto_mapper <- function(model,
                                        train_dl,
                                        valid_dl = NULL,
                                        epochs   = 50,
                                        lr       = 1e-3,
                                        loss_kind = c("mse", "bce_logits"),
                                        beta      = 1.0,
                                        beta_schedule = NULL,
                                        grad_clip = NULL,
                                        device = NULL,
                                        verbose = TRUE,
                                        patience = NULL) {

  loss_kind <- match.arg(loss_kind)

  # Device
  if (is.null(device)) {
    device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  }

  # --------- IMPORTANT: build the model ONCE before creating the optimizer ----
  # Peek one batch from the train dataloader
  it <- dataloader_make_iter(train_dl)
  b0 <- dataloader_next(it)
  x0 <- if (is.list(b0)) b0$x else b0
  if (!inherits(x0, "torch_tensor")) stop("fit(): could not infer a tensor batch from train_dl.")

  # Move a tiny batch to device and force lazy build
  x0 <- x0$to(device = device)
  if (is.function(model$.ensure_built)) {
    # call the internal builder directly to avoid burning a forward
    model$.ensure_built(x0)
  } else {
    # fallback: run a single dry forward
    with_no_grad({ model(x0, sample = FALSE) })
  }

  # Now move the (built) model to device
  model$to(device = device)

  # Sanity check: ensure there are parameters
  if (length(model$parameters) == 0) {
    stop("fit(): model has no parameters after build. Check that encoder/decoder were created.")
  }

  # Optimizer
  opt <- optim_adam(model$parameters, lr = lr)

  # History
  hist <- data.frame(
    epoch = integer(),
    train_loss = numeric(),
    train_rec  = numeric(),
    train_kl   = numeric(),
    valid_loss = numeric(),
    valid_rec  = numeric(),
    valid_kl   = numeric(),
    stringsAsFactors = FALSE
  )

  best_val <- Inf
  best_state <- NULL
  patience_counter <- 0

  train_epoch <- function(ep) {
    model$train()
    running_total <- 0; running_rec <- 0; running_kl <- 0; n_batches <- 0
    beta_t <- if (!is.null(beta_schedule)) beta_schedule(ep) else beta

    coro::loop(for (b in train_dl) {
      if (is.list(b)) {
        x <- b$x$to(device = device)
        y <- (if (!is.null(b$y)) b$y else b$x)$to(device = device)
        mask <- if (!is.null(b$mask)) b$mask$to(device = device) else NULL
      } else {
        x <- b$to(device = device); y <- x; mask <- NULL
      }

      opt$zero_grad()
      out <- model(x, sample = TRUE)
      rec <- masked_reconstruction_loss(out$recon, y, mask = mask, kind = loss_kind)
      kld <- kl_standard_normal(out$mu, out$logvar)
      loss <- rec + beta_t * kld

      loss$backward()
      if (!is.null(grad_clip)) nn_utils_clip_grad_norm_(model$parameters, max_norm = grad_clip)
      opt$step()

      running_total <- running_total + as.numeric(loss$item())
      running_rec   <- running_rec   + as.numeric(rec$item())
      running_kl    <- running_kl    + as.numeric(kld$item())
      n_batches     <- n_batches + 1
    })

    c(total = running_total / max(1, n_batches),
      rec   = running_rec   / max(1, n_batches),
      kl    = running_kl    / max(1, n_batches))
  }

  valid_epoch <- function(ep) {
    if (is.null(valid_dl)) return(c(total = NA_real_, rec = NA_real_, kl = NA_real_))
    model$eval()
    running_total <- 0; running_rec <- 0; running_kl <- 0; n_batches <- 0
    beta_t <- if (!is.null(beta_schedule)) beta_schedule(ep) else beta

    coro::loop(for (b in valid_dl) {
      if (is.list(b)) {
        x <- b$x$to(device = device)
        y <- (if (!is.null(b$y)) b$y else b$x)$to(device = device)
        mask <- if (!is.null(b$mask)) b$mask$to(device = device) else NULL
      } else {
        x <- b$to(device = device); y <- x; mask <- NULL
      }

      with_no_grad({
        out <- model(x, sample = FALSE)
        rec <- masked_reconstruction_loss(out$recon, y, mask = mask, kind = loss_kind)
        kld <- kl_standard_normal(out$mu, out$logvar)
        loss <- rec + beta_t * kld
      })

      running_total <- running_total + as.numeric(loss$item())
      running_rec   <- running_rec   + as.numeric(rec$item())
      running_kl    <- running_kl    + as.numeric(kld$item())
      n_batches     <- n_batches + 1
    })

    c(total = running_total / max(1, n_batches),
      rec   = running_rec   / max(1, n_batches),
      kl    = running_kl    / max(1, n_batches))
  }

  for (ep in seq_len(epochs)) {
    tr <- train_epoch(ep)
    va <- valid_epoch(ep)

    hist <- rbind(hist, data.frame(
      epoch      = ep,
      train_loss = tr["total"], train_rec = tr["rec"], train_kl = tr["kl"],
      valid_loss = va["total"], valid_rec = va["rec"], valid_kl = va["kl"]
    ))

    if (verbose) {
      message(sprintf(
        "[%03d/%03d] train: loss=%.5f rec=%.5f kl=%.5f | valid: loss=%s rec=%s kl=%s\n",
        ep, epochs,
        tr["total"], tr["rec"], tr["kl"],
        ifelse(is.na(va["total"]), "NA", sprintf("%.5f", va["total"])),
        ifelse(is.na(va["rec"]),   "NA", sprintf("%.5f", va["rec"])),
        ifelse(is.na(va["kl"]),    "NA", sprintf("%.5f", va["kl"]))
      ))
    }

    current_val <- va["total"]
    if (!is.na(current_val) && is.finite(current_val)) {
      if (current_val < best_val) {
        best_val <- current_val
        best_state <- model$state_dict()
        patience_counter <- 0
      } else {
        patience_counter <- patience_counter + 1
      }
    }

    if (!is.null(patience) && patience_counter >= patience) {
      if (verbose) message("Early stopping triggered at epoch", ep, "\n")
      break
    }
  }

  list(
    model = model,
    history = hist,
    best_val = best_val,
    best_state = best_state,
    load_best = function() {
      if (!is.null(best_state)) model$load_state_dict(best_state)
      invisible(NULL)
    },
    encode = function(X, device_override = NULL, sample = FALSE) {
      dev <- if (is.null(device_override)) device else device_override
      model$eval(); with_no_grad({ X <- X$to(device = dev); model$encode(X, sample = sample)$z })
    },
    reconstruct = function(X, device_override = NULL, sample = FALSE) {
      dev <- if (is.null(device_override)) device else device_override
      model$eval(); with_no_grad({ X <- X$to(device = dev); model(X, sample = sample)$recon })
    }
  )
}


#' @keywords internal
train_variational_mapper <- function(
    X,                    # torch tensor [N, ...]
    Y = NULL,             # optional target tensor [N, ...]; if NULL, autoencodes X
    mask = NULL,          # optional mask [N, ...] (1=use, 0=ignore)
    # Model params
    latent_dim = 32,
    enc_hidden = c(512, 256),
    dec_hidden = c(256, 512),
    p_drop = 0.1,
    out_kind = c("linear", "tanh", "sigmoid", "logits"),
    target_shape = NULL,  # if NULL and Y provided -> inferred from Y; else from X
    # Training params
    epochs = 30,
    batch_size = 64,
    lr = 1e-3,
    loss_kind = c("mse", "bce_logits"),
    beta = 1.0,
    beta_warmup = 0,      # integer epochs; if >0, linear warm-up to beta
    grad_clip = NULL,
    valid_split = 0.2,
    shuffle = TRUE,
    device = NULL,
    verbose = TRUE,
    patience = NULL,
    seed = 42
) {
  stopifnot(inherits(X, "torch_tensor"))
  N <- as.integer(X$size(1))
  loss_kind <- match.arg(loss_kind)
  out_kind  <- match.arg(out_kind)

  if (!is.null(seed)) {
    set.seed(seed)          # base R RNG
    torch_manual_seed(seed) # torch CPU + CUDA RNG
  }

  # Determine target shape (if not provided)
  if (is.null(target_shape)) {
    if (!is.null(Y)) {
      target_shape <- as.integer(Y$size())[-1]
    } else {
      target_shape <- as.integer(X$size())[-1]
    }
  } else {
    target_shape <- as.integer(target_shape)
  }

  # Split train/valid indices
  idx <- seq_len(N)
  if (shuffle) idx <- sample(idx)
  n_valid <- max(0L, round(valid_split * N))
  valid_idx <- if (n_valid > 0) idx[seq_len(n_valid)] else integer(0)
  train_idx <- if (n_valid > 0) idx[-seq_len(n_valid)] else idx
  if (length(train_idx) == 0) stop("No training samples after split; reduce valid_split.")

  # Slice tensors helper
  take_rows <- function(TT, ids) if (is.null(TT)) NULL else TT[ids,..]

  X_tr <- take_rows(X, train_idx); X_va <- take_rows(X, valid_idx)
  Y_tr <- take_rows(Y, train_idx); Y_va <- take_rows(Y, valid_idx)
  M_tr <- take_rows(mask, train_idx); M_va <- take_rows(mask, valid_idx)

  ds_tr <- vam_dataset(X_tr, Y_tr, M_tr)
  dl_tr <- dataloader(ds_tr, batch_size = batch_size, shuffle = TRUE)

  dl_va <- NULL
  if (length(valid_idx) > 0) {
    ds_va <- vam_dataset(X_va, Y_va, M_va)
    dl_va <- dataloader(ds_va, batch_size = batch_size, shuffle = FALSE)
  }

  # Instantiate model
  model <- variational_auto_mapper(
    latent_dim = latent_dim,
    enc_hidden = enc_hidden,
    dec_hidden = dec_hidden,
    p_drop = p_drop,
    out_kind = out_kind,
    target_shape = target_shape
  )

  # beta schedule (optional linear warm-up)
  beta_schedule <- NULL
  if (!is.null(beta_warmup) && beta_warmup > 0) {
    beta_schedule <- function(ep) {
      x <- min(1, ep / beta_warmup)
      x * beta
    }
  }

  # Fit
  fit <- fit_variational_auto_mapper(
    model = model,
    train_dl = dl_tr,
    valid_dl = dl_va,
    epochs = epochs,
    lr = lr,
    loss_kind = loss_kind,
    beta = beta,
    beta_schedule = beta_schedule,
    grad_clip = grad_clip,
    device = device,
    verbose = verbose,
    patience = patience
  )

  param_count <- sum(sapply(fit$model$parameters, function(p) p$numel()))

  desc <- paste0(
    "Variational Auto-Mapper (VAM). Latent dimension: ", latent_dim, ". Encoder hidden layers: ", paste(enc_hidden, collapse = " -> "), ". Decoder hidden layers: ", paste(dec_hidden, collapse = " -> "), ". Dropout: ", p_drop, ". Output kind: ", out_kind, ". Target shape: [", paste(target_shape, collapse = " x "), "]. Total parameters: ", format(param_count, big.mark = ",")
  )

  # Return a compact handle with convenience helpers (base tensors in/out)
  structure(list(
    desc = desc,
    model = fit$model,
    history = fit$history,
    best_val = fit$best_val,
    load_best = fit$load_best,
    encode = fit$encode,
    reconstruct = fit$reconstruct,
    decode_from_latent = fit$decode_from_latent,
    predict = function(X_new, n_samples = 1, sample = TRUE, device_override = NULL) {
      dev <- if (is.null(device_override)) {
        if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
      } else device_override

      fit$model$eval()
      with_no_grad({
        Xn <- X_new$to(device = dev)

        if (n_samples <= 1) {stop("need at least one sample")}

        # Multiple samples: draw S independent passes
        recons <- vector("list", n_samples)
        latents <- vector("list", n_samples)
        for (s in seq_len(n_samples)) {
          out <- fit$model(Xn, sample = sample)
          recons[[s]] <- out$recon
          latents[[s]] <- out$z
        }
        # Stack on a new leading dimension [S, ...]
        recons_s <- torch_stack(recons, dim = 1L)
        latents_s <- torch_stack(latents, dim = 1L)
        list(recon = recons_s, latent = latents_s)
      })
    },

    predict_mean = function(X_new, n_samples = 20, device_override = NULL) {
      # Convenience: MC mean & std over samples.
      # Returns list(mean_recon, std_recon, mean_latent, std_latent)
      dev <- if (is.null(device_override)) {
        if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
      } else device_override

      fit$model$eval()
      with_no_grad({
        Xn <- X_new$to(device = dev)

        # Collect S samples
        recons <- vector("list", n_samples)
        latents <- vector("list", n_samples)
        for (s in seq_len(n_samples)) {
          out <- fit$model(Xn, sample = TRUE)
          recons[[s]] <- out$recon
          latents[[s]] <- out$z
        }
        R <- torch_stack(recons, dim = 1L)   # [S, B, ...]
        Z <- torch_stack(latents, dim = 1L)  # [S, B, D]

        mean_recon <- torch_mean(R, dim = 1L)
        std_recon  <- torch_std(R,  dim = 1L, unbiased = FALSE)
        mean_lat   <- torch_mean(Z, dim = 1L)
        std_lat    <- torch_std(Z,  dim = 1L, unbiased = FALSE)

        list(
          mean_recon = mean_recon, std_recon = std_recon,
          mean_latent = mean_lat,  std_latent = std_lat
        )
      })
    },
    params = list(
      latent_dim = latent_dim, enc_hidden = enc_hidden, dec_hidden = dec_hidden,
      p_drop = p_drop, out_kind = out_kind, target_shape = target_shape,
      epochs = epochs, batch_size = batch_size, lr = lr, loss_kind = loss_kind,
      beta = beta, beta_warmup = beta_warmup, grad_clip = grad_clip,
      valid_split = valid_split, seed = seed
    )
  ), class = "vam_fit")
}


#' @keywords internal
plot_graph <- function(ts, pred_funs, alpha = 0.05, dates = NULL, line_size = 1.3, label_size = 11,
                       forcat_band = "seagreen2", forcat_line = "seagreen4", hist_line = "gray43",
                       label_x = "Horizon", label_y= "Forecasted Var", date_format = "%b-%Y")
{
  preds <- Reduce(rbind, map(pred_funs, ~ quantile(.x$rfun(1000), probs = c(alpha, 0.5, (1-alpha)))))
  if(length(pred_funs)==1){preds <- matrix(preds, nrow = 1)}
  colnames(preds) <- c("lower", "median", "upper")
  future <- nrow(preds)

  if(is.null(dates)){x_hist <- 1:length(ts)} else {x_hist <- as.Date(as.character(dates))}
  if(is.null(dates)){x_forcat <- length(ts) + 1:nrow(preds)} else {x_forcat <- as.Date(as.character(tail(dates, 1)))+ 1:future}

  forecast_data <- data.frame(x_forcat = x_forcat, preds)
  historical_data <- data.frame(x_all = as.Date(c(x_hist, x_forcat)), y_all = c(ts = ts, pred = preds[, "median"]))

  plot <- ggplot()+ geom_line(data = historical_data, aes(x = .data$x_all, y = .data$y_all), color = hist_line, linewidth = line_size)
  plot <- plot + geom_ribbon(data = forecast_data, aes(x = x_forcat, ymin = .data$lower, ymax = .data$upper), alpha = 0.3, fill = forcat_band)
  plot <- plot + geom_line(data = forecast_data, aes(x = x_forcat, y = median), color = forcat_line, linewidth = line_size)
  if(!is.null(dates)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_labels = date_format)}
  if(is.null(dates)){plot <- plot + scale_x_continuous(name = paste0("\n", label_x))}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)
  plot <- plot + ylab(label_y) + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}


#' @keywords internal
dts <- function(ts, lag = 1)
{
  scaled_ts <- tail(ts, -lag)/head(ts, -lag)-1
  scaled_ts[!is.finite(scaled_ts)] <- NA
  if(anyNA(ts)){scaled_ts <- na_kalman(scaled_ts)}
  return(scaled_ts)
}


#' @keywords internal
smart_reframer <- function(ts, seq_len, stride)
{
  n_length <- length(ts)
  if(seq_len > n_length | stride > n_length){stop("vector too short for sequence length or stride")}
  if(n_length%%seq_len > 0){ts <- tail(ts, - (n_length%%seq_len))}
  n_length <- length(ts)
  idx <- seq(from = 1, to = (n_length - seq_len + 1), by = 1)
  reframed <- t(sapply(idx, function(x) ts[x:(x+seq_len-1)]))
  if(seq_len == 1){reframed <- t(reframed)}
  idx <- rev(seq(nrow(reframed), 1, - stride))
  reframed <- reframed[idx,,drop = FALSE]
  colnames(reframed) <- paste0("t", 1:seq_len)
  return(reframed)
}

#' @keywords internal
gmix <- function(x,
                 K_max = 10,     # upper bound on clusters
                 seed   = 1,     # k-means reproducibility
                 ...) {          # extra args to kmeans()
  stopifnot(is.numeric(x), is.vector(x))
  n <- length(x)

  K_pos <- length(unique(x))
  K_eff <- min(K_max, K_pos)

  if (K_eff < K_max) warning("Reduced k from ", K_max, " to ", K_eff,
                             " because of only ", K_pos, " distinct points.")

  ## -------------------------------------------------------------- ##
  ## 1.  Compute WSS for k = 1 ... K_max ----------------------------- ##
  ## -------------------------------------------------------------- ##
  wss <- numeric(K_eff)
  set.seed(seed)
  for (k in 1:K_eff) {
    wss[k] <- kmeans(x, centers = k, ...)$tot.withinss
  }

  ## -------------------------------------------------------------- ##
  ## 2.  Detect the elbow (max curvature) ------------------------- ##
  ## ----------- --------------------------------------------------- ##
  # First & second finite differences: deltaWSS(k) = WSS(k-1) - WSS(k)
  d1  <- -diff(wss)                  # length K_eff-1
  d2  <- diff(d1)                    # length K_eff-2
  k_hat <- which.max(d2) + 1L        # add 1 because d2 starts at k = 3

  ## Fallback if curvature is flat (rare): default to 2 clusters
  if (length(k_hat) == 0 || is.na(k_hat) || k_hat < 2) k_hat <- 2

  ## -------------------------------------------------------------- ##
  ## 3.  Final k-means with k = k^ -------------------------------- ##
  ## -------------------------------------------------------------- ##
  set.seed(seed)
  km <- kmeans(x, centers = k_hat, ...)

  clusters <- km$cluster
  centers  <- as.numeric(km$centers)
  sizes    <- km$size
  weights  <- sizes / n

  # Component SDs (protect 1-point clusters with a small positive value)
  sigmas <- vapply(split(x, clusters), function(v)
    if (length(v) > 1) sd(v) else 1e-8, numeric(1))

  ## -------------------------------------------------------------- ##
  ## 4.  Mixture helpers ------------------------------------------ ##
  ## -------------------------------------------------------------- ##
  pdf_fun <- function(z)
    vapply(z, \(zz) sum(weights * dnorm(zz, centers, sigmas)), numeric(1))

  cdf_fun <- function(z)
    vapply(z, \(zz) sum(weights * pnorm(zz, centers, sigmas)), numeric(1))

  ## --- 1.  Build a grid that covers the tails -----------------------
  grid_n   <- 2000                               # resolution (tweak if needed)
  tail_pad <- 6 * max(sigmas)                    # same padding idea as before
  x_grid   <- seq(min(x) - tail_pad,
                  max(x) + tail_pad,
                  length.out = grid_n)

  cdf_grid <- cdf_fun(x_grid)

  ## Remove duplicates (can happen in flat tails) ---------------------
  ix <- !duplicated(cdf_grid)
  cdf_u <- cdf_grid[ix]
  x_u   <- x_grid[ix]

  ## --- 2.  Monotone Hermite spline: probability -> quantile --------
  q_spline <- splinefun(cdf_u, x_u, method = "monoH.FC")

  ## --- 3.  Final inverse-CDF ---------------------------------------
  icdf_fun <- function(p) {
    stopifnot(all(p >= 0 & p <= 1))

    ## exact edges
    p[p == 0] <- min(cdf_u)
    p[p == 1] <- max(cdf_u)

    ## for probabilities outside the pre-computed range
    p[p < min(cdf_u)] <- min(cdf_u)
    p[p > max(cdf_u)] <- max(cdf_u)

    q_spline(p)
  }

  sampler_fun <- function(n) {
    g <- sample(seq_along(weights), n, TRUE, prob = weights)
    rnorm(n, centers[g], sigmas[g])
  }

  pred_funs <- list(rfun = sampler_fun, dfun = pdf_fun, pfun = cdf_fun, qfun = icdf_fun)

  attr(pred_funs, "kmeans") <- km
  attr(pred_funs, "weights") <- weights
  attr(pred_funs, "centers") <- centers
  attr(pred_funs, "sigmas") <- sigmas
  attr(pred_funs, "wss") <- wss
  attr(pred_funs, "k_hat") <- k_hat

  return(pred_funs)
}
