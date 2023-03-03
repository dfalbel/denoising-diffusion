box::use(torch[...])
box::use(torchvision[...])
box::use(torchdatasets[...])

dir <- "./data" #caching directory

diffusion_dataset <- dataset(
  "DiffusionDataset",
  initialize = function(dataset, image_size, ...) {

    self$image_size <- image_size

    self$transform <- function(x) {
      x |>
        transform_to_tensor() |>
        transform_resize(self$image_size)
    }

    self$data <- dataset(
      dir,
      transform = self$transform,
      ...
    )
  },
  .getitem = function(i) {
    self$data[i]
  },
  .length = function() {
    length(self$data)
  }
)

debug_dataset <- dataset(
  initialize = function(dataset) {
    self$dataset <- dataset
  },
  .getitem = function(i) {
    self$dataset$.getitem((i-1) %% 2 + 1)
  },
  .length = function() {
    length(self$dataset)
  }
)

make_dataset <- function(type = c("pets", "flowers", "debug"), image_size) {
  type <- rlang::arg_match(type)
  if (type == "pets") {
    diffusion_dataset(
      torchdatasets::oxford_pet_dataset,
      target_type = "species",
      image_size,
      split = "train",
      download = TRUE
    )
  } else if (type == "flowers") {
    diffusion_dataset(
      torchdatasets::oxford_flowers102_dataset,
      image_size,
      split = c("train", "test"),
      download = TRUE
    )
  } else if (type == "debug") {
    debug_dataset(diffusion_dataset(
      torchdatasets::oxford_pet_dataset,
      target_type = "species",
      image_size,
      split = "train",
      download = TRUE
    ))
  } else {
    cli::cli_abort("Unsupported dataset type {.val {type}}")
  }
}
