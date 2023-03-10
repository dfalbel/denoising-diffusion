inception_encoder <- torch::nn_module(
  initialize = function(kid_image_size) {
    self$inception <- torchvision::model_inception_v3(pretrained = TRUE)
    self$inception$fc <- torch::nn_identity()
    self$inception$eval()
    self$kid_image_size <- kid_image_size
  },
  forward = function(x) {
    x |>
      torchvision::transform_resize(self$kid_image_size) |>
      torchvision::transform_normalize(
        mean=c(0.485, 0.456, 0.406),
        std=c(0.229, 0.224, 0.225)
      ) |>
      self$inception()
  }
)

metric_kid_base <- luz_metric(
  "kid",
  abbrev = "kid",
  initialize = function(kid_img_size = c(299, 299)) {
    # a pretrained InceptionV3 is used without its classification layer
    self$encoder <- inception_encoder(kid_img_size)
    self$encoder$eval()
    self$kid <- 0
    self$step <- 0
  },
  polynomial_kernel = function(features_1, features_2) {
    feature_dimensions <- features_1$shape[2]
    (torch_matmul(features_1, features_2$t()) / feature_dimensions + 1.0) ^ 3.0
  },
  update = function(real_images, generated_images, sample_weight = NULL) {
    self$encoder$to(device=ctx$device)
    real_features <- self$encoder(real_images)
    generated_features <- self$encoder(generated_images)

    # compute polynomial kernels using the two sets of features
    kernel_real <- self$polynomial_kernel(real_features, real_features)
    kernel_generated <- self$polynomial_kernel(generated_features, generated_features)
    kernel_cross <- self$polynomial_kernel(real_features, generated_features)

    # estimate the squared maximum mean discrepancy using the average kernel values
    batch_size <- real_features$shape[1]
    mean_kernel_real <- torch_sum(kernel_real * (1.0 - torch_eye(batch_size, device=ctx$device))) /
      (batch_size * (batch_size - 1.0))

    mean_kernel_generated <- torch_sum(kernel_generated * (1.0 - torch_eye(batch_size, device=ctx$device))) /
      (batch_size * (batch_size - 1.0))

    mean_kernel_cross <- torch_mean(kernel_cross)
    kid <- mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

    self$step <- self$step + 1
    self$kid <- self$kid + kid$item()
  },
  compute = function() {
    self$kid / self$step
  }
)

metric_kid <- luz_metric(
  "kid_metric",
  inherit = metric_kid_base,
  initialize = function(diffusion_steps = 5) {
    super$initialize()
    self$diffusion_steps <- diffusion_steps
  },
  update = function(preds, target) {
    ctx$model$eval()
    with_no_grad({
      images <- ctx$model$normalize$denormalize(ctx$input)
      generated_images <- ctx$model$generate(
        num_images=images$shape[1],
        diffusion_steps=self$diffusion_steps
      )
      super$update(images, generated_images)
    })
    ctx$model$train()
  }
)
