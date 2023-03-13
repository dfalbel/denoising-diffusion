box::use(torch[...])
box::use(torchvision[...])
box::use(torchdatasets[...])
box::use(zeallot[...])

dir <- "./data" #caching directory

diffusion_dataset <- dataset(
  "DiffusionDataset",
  initialize = function(dataset, image_size, ...) {

    self$image_size <- image_size

    self$transform <- function(x) {
      img <- x |>
        transform_to_tensor()

      c(ch, height, width) %<-% img$size()
      crop_size <- min(height, width)

      img |>
        transform_center_crop(c(crop_size, crop_size)) |>
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
    train_dataset <- diffusion_dataset(
      torchdatasets::oxford_pet_dataset,
      target_type = "species",
      image_size,
      split = "train",
      download = TRUE
    )
    valid_dataset <- diffusion_dataset(
      torchdatasets::oxford_pet_dataset,
      target_type = "species",
      image_size,
      split = "valid",
      download = TRUE
    )
    list(train_dataset, valid_dataset)
  } else if (type == "flowers") {
    dataset <- diffusion_dataset(
      torchdatasets::oxford_flowers102_dataset,
      image_size,
      split = c("train", "test", "valid"),
      download = TRUE
    )
    train_indexes <- sample.int(length(dataset), 0.8*length(dataset))
    valid_indexes <- seq_len(length(dataset))[-train_indexes]

    list(
      dataset_repeat(dataset_subset(dataset, train_indexes)),
      dataset_repeat(dataset_subset(dataset, valid_indexes))
    )
  } else if (type == "debug") {
    train_dataset <- debug_dataset(diffusion_dataset(
      torchdatasets::oxford_pet_dataset,
      target_type = "species",
      image_size,
      split = "train",
      download = TRUE
    ))
    valid_dataset <- debug_dataset(diffusion_dataset(
      torchdatasets::oxford_pet_dataset,
      target_type = "species",
      image_size,
      split = "valid",
      download = TRUE
    ))
    list(train_dataset, valid_dataset)
  } else {
    cli::cli_abort("Unsupported dataset type {.val {type}}")
  }
}

dataset_repeat <- dataset(
  initialize = function(dataset, repeats = 5) {
    self$dataset <- dataset
    self$repeats <- repeats
    self$data_length <- length(dataset)
  },
  .getitem = function(i) {
    index <- i %% self$data_length + 1
    self$dataset[index]
  },
  .length = function() {
    self$repeats * self$data_length
  }
)

