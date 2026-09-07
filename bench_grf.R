# R grf side of the cross-framework forest benchmark.
# Reads the datasets written by benchmark.GrfBenchmark and appends rows to bench_results.csv.
#
#   grf::causal_forest (Y.hat/W.hat disabled) -> group "raw-causal"
#   grf::causal_forest (default local centering) -> group "centered-causal"
#   grf::regression_forest                     -> group "regression"
#
# Run:  R_LIBS_USER=$HOME/Library/R/arm64/4.2/library-grf Rscript bench_grf.R

suppressPackageStartupMessages(library(grf))

data_dir <- "data"

N_TRAINS <- c(5000, 20000)
N_TREES  <- 500
MAX_DEPTH <- 30
MIN_LEAF <- 5
MTRY <- 10
SAMPLE_FRAC <- 0.5
HONEST_FRAC <- 0.5
REPS <- 3
SEED <- 42
results <- file.path(data_dir, "bench_results.csv")

load_xy <- function(path) {
  df <- read.csv(path)
  xcols <- grep("^x", names(df), value = TRUE)
  list(X = as.matrix(df[, xcols]), T = df$treatment, Y = df$label, tau = df$true_hte)
}

rmse <- function(a, b) sqrt(mean((a - b)^2))

append_row <- function(model, group, n_train, rep, train_ms, predict_ms, err) {
  line <- sprintf("r-grf,%s,%s,%d,%d,%d,%.3f,%.3f,%.5f",
                  model, group, n_train, N_TREES, rep, train_ms, predict_ms, err)
  cat(line, "\n", file = results, append = TRUE, sep = "")
  cat(sprintf("  [%-24s n=%6d rep=%d] train=%10.1f ms  predict=%8.1f ms  rmse=%.4f\n",
              model, n_train, rep, train_ms, predict_ms, err))
}

run_cf <- function(X, W, Y, Xte, centered) {
  args <- list(X = X, Y = Y, W = W, num.trees = N_TREES, sample.fraction = SAMPLE_FRAC,
               mtry = MTRY, min.node.size = MIN_LEAF, honesty = TRUE,
               honesty.fraction = HONEST_FRAC, ci.group.size = 1, num.threads = 0, seed = SEED)
  if (!centered) { args$Y.hat <- rep(mean(Y), length(Y)); args$W.hat <- rep(mean(W), length(W)) }
  t0 <- proc.time()[["elapsed"]]
  f  <- do.call(causal_forest, args)
  tr <- (proc.time()[["elapsed"]] - t0) * 1000
  t0 <- proc.time()[["elapsed"]]
  p  <- predict(f, Xte)$predictions
  pr <- (proc.time()[["elapsed"]] - t0) * 1000
  list(tr = tr, pr = pr, pred = p)
}

run_rf <- function(X, Y, Xte) {
  t0 <- proc.time()[["elapsed"]]
  f  <- regression_forest(X, Y, num.trees = N_TREES, sample.fraction = SAMPLE_FRAC,
                          mtry = MTRY, min.node.size = MIN_LEAF, honesty = TRUE,
                          honesty.fraction = HONEST_FRAC, ci.group.size = 1,
                          num.threads = 0, seed = SEED)
  tr <- (proc.time()[["elapsed"]] - t0) * 1000
  t0 <- proc.time()[["elapsed"]]
  p  <- predict(f, Xte)$predictions
  pr <- (proc.time()[["elapsed"]] - t0) * 1000
  list(tr = tr, pr = pr, pred = p)
}

te <- load_xy(file.path(data_dir, "test.csv"))
mean_te <- te$X[, 1] + 0.5 * te$X[, 2] - 1.0 * te$X[, 3]

cat(sprintf("grf %s  nTrees=%d maxDepth(n/a) minLeaf=%d mtry=%d sampleFrac=%.2f\n\n",
            as.character(packageVersion("grf")), N_TREES, MIN_LEAF, MTRY, SAMPLE_FRAC))

for (n in N_TRAINS) {
  cat(sprintf("======== N_train = %d ========\n", n))
  tr_data <- load_xy(file.path(data_dir, sprintf("train_%d.csv", n)))
  invisible(run_rf(tr_data$X[1:200, ], tr_data$Y[1:200], te$X[1:10, ]))  # warm-up
  for (rep in 1:REPS) {
    r <- run_cf(tr_data$X, tr_data$T, tr_data$Y, te$X, centered = FALSE)
    append_row("causal_forest (raw)", "raw-causal", n, rep, r$tr, r$pr, rmse(r$pred, te$tau))

    r <- run_cf(tr_data$X, tr_data$T, tr_data$Y, te$X, centered = TRUE)
    append_row("causal_forest (centered)", "centered-causal", n, rep, r$tr, r$pr, rmse(r$pred, te$tau))

    r <- run_rf(tr_data$X, tr_data$Y, te$X)
    append_row("regression_forest", "regression", n, rep, r$tr, r$pr, rmse(r$pred, mean_te))
  }
}

cat(sprintf("\nAppended to %s\n", results))
