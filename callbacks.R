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
    images <- ctx$model$reverse_diffusion(self$initial_noise, self$diffusion_steps)
    images <- images$permute(c(1,3,4,2))$to(dtype = torch_float(), device = "cpu")
    log_event(
      imgs = summary_image(as.array(images), step = ctx$epoch)
    )
  }
)
