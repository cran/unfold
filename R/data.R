#'Tech Stock Time Series Dataset
#'
#' A multivariate dataset for closing prices for several major tech stocks over time. Source: YahooFinance.
#'
#' @format A data frame with 2133 observations of 4 variables:
#' \describe{
#'   \item{dates}{Character vector of dates in "YYYY-MM-DD" format.}
#'   \item{TSLA.Close}{Numeric. Closing prices for Tesla.}
#'   \item{MSFT.Close}{Numeric. Closing prices for Microsoft.}
#'   \item{MARA.Close}{Numeric. Closing prices for MARA Holdings.}
#' }
#'
#' @usage data(dummy_set)
#'
#' @examples
#' data(dummy_set)
#' plot(as.Date(dummy_set$dates), dummy_set$TSLA.Close, type = "l")
"dummy_set"
