# R/utils_paths.R

suppressPackageStartupMessages({
  library(here)
  library(config)
})

get_project_config <- function() {
  config::get()
}

get_path <- function(name) {
  cfg <- get_project_config()
  if (!("paths" %in% names(cfg))) {
    stop("Config file has no 'paths' section.")
  }
  if (!(name %in% names(cfg$paths))) {
    stop(sprintf("Path '%s' not defined in config.yml", name))
  }
  here::here(cfg$paths[[name]])
}

raw_dir <- function()          get_path("raw")
intermediate_dir <- function() get_path("intermediate")
clean_dir <- function()        get_path("clean")
models_dir <- function()       get_path("models")
metadata_dir <- function()     get_path("metadata")

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  invisible(path)
}

