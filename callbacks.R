box::use(luz[...])
box::use(torch[...])
box::use(tfevents[...])

callback_generate_samples <- luz_callback(
  name = "GenerateSamples",
  initialize = function(num_images = 20, diffusion_steps = 20) {
    self$num_images <- num_images
    self$diffusion_steps <- diffusion_steps
  },
  on_train_begin = function() {
    self$initial_noise <- torch_randn(
      c(self$num_images, ctx$model$image_size),
      device = ctx$accelerator$device
    )
  },
  on_epoch_end = function () {
    ctx$model$eval()
    with_no_grad({
      images <- ctx$model$reverse_diffusion(self$initial_noise, self$diffusion_steps)
      plot_tensors(images, identity)
      images <- images$permute(c(1,3,4,2))$to(dtype = torch_float(), device = "cpu")
    })
    log_event(
      imgs = summary_image(as.array(images), step = ctx$epoch)
    )

    ctx$model$train()
  }
)

image_loss <- luz_metric(
  "image_loss",
  abbrev = "image_loss",
  inherit = luz_metric_mae,
  update = function(preds, target) {
    super$update(preds$pred_images, ctx$input)
  }
)

inception_encoder <- torch::nn_module(
  initialize = function(kid_image_size = c(75, 75)) {
    self$inception <- torchvision::model_inception_v3(pretrained = TRUE)
    self$inception$fc <- torch::nn_identity()
    self$kid_image_size <- kid_image_size
    self$transform_normalize <- function(x) {
      torchvision::transform_normalize(
        x,
        mean = c(0.485, 0.456, 0.406),
        std = c(0.229, 0.224, 0.225)
      )
    }
  },
  forward = function(x) {
    x |>
      self$transform_normalize() |>
      torchvision::transform_resize(self$kid_image_size, interpolation = 2) |> # minimum size accepted by the inception net
      self$inception()
  }
)

metric_kid <- luz_metric(
  "kid",
  abbrev = "kid",
  initialize = function(kid_img_size = c(75, 75)) {
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

metric_kid_wrapper <- luz_metric(
  "kid_metric",
  inherit = metric_kid,
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

plot_tensors <- function(x, denormalize = identity, ncol = 6) {
  if (inherits(x, "torch_tensor"))
    x <- torch::torch_unbind(x)

  nrow <- ceiling(length(x) / ncol)
  graphics::par(mfrow=c(nrow,ncol), mai = rep(0.01, 4))

  for (img in x) {
    img |>
      denormalize() |>
      torch::torch_clip(0, 1) |>
      (\(x) x$cpu())() |>
      as.array() |>
      aperm(c(2,3,1)) |>
      grDevices::as.raster() |>
      plot(asp = NA)
  }
}

