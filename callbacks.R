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

