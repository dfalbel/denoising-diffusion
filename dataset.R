box::use(torch[...])
box::use(torchvision[...])
box::use(torchdatasets[...])

dir <- "~/Documents/posit/denoising-diffusion/pets" #caching directory

# A light wrapper around the `oxford_pet_dataset` that resizes and transforms
# input images and masks to the specified `size` and introduces the `augmentation`
# argument, allowing us to specify transformations that must be synced between
# images and masks, eg. flipping, cropping, etc.
pet_dataset <- torch::dataset(
  inherit = oxford_pet_dataset,
  initialize = function(..., augmentation = NULL, size = c(224, 224)) {

    input_transform <- function(x) {
      x %>%
        transform_to_tensor() %>%
        transform_resize(size)
    }

    super$initialize(
      ...,
      transform = input_transform
    )

    if (is.null(augmentation))
      self$augmentation <- function(...) {list(...)}
    else
      self$augmentation <- augmentation

  },
  .getitem = function(i) {
    items <- super$.getitem(i)
    list(items[[1]], 1)
  }
)

train_ds <- pet_dataset(
  dir,
  download = TRUE,
  split = "train",
  size = c(64, 64)
)

valid_ds <- pet_dataset(
  dir,
  download = TRUE,
  split = "valid",
  size = c(64, 64)
)
