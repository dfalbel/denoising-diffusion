box::use(./unet[unet, swish])
box::use(torch[...])
box::use(zeallot[...])

diffusion_schedule <- nn_module(
  initialize = function(min_signal_rate = 0.02, max_signal_rate = 0.95) {
    self$start_angle <- nn_buffer(torch_acos(max_signal_rate))
    self$end_angle <- nn_buffer(torch_acos(min_signal_rate))
  },
  forward = function(diffusion_times) {
    angles <- self$start_angle + diffusion_times*(self$end_angle - self$start_angle)

    list(
      signal = torch_cos(angles),
      noise = torch_sin(angles)
    )
  }
)

normalize <- nn_module(
  initialize = function(num_channels, num_steps = 20) {
    self$means <- nn_buffer(torch_zeros(1, num_channels, 1, 1))
    self$stds <- nn_buffer(torch_zeros(1, num_channels, 1, 1))
    self$step <- 0
    self$num_steps <- num_steps
  },
  forward = function(x) {

    if (self$step < self$num_steps) {
      self$step <- self$step + 1
      self$update_stats(x)
    }

    (x - self$means) / self$stds
  },
  update_stats = function(x) {
    means <- torch_mean(x, dim = c(1,3,4), keepdim = TRUE)
    vars <- torch_var(x, dim = c(1,3,4), keepdim = TRUE)

    w <- 1 / self$step

    self$means$mul_(1-w)$add_(w*means)
    self$stds$pow_(2)$mul_(1-w)$add_(w*vars)$sqrt_()
  },
  denormalize = function(x) {
    x * self$stds + self$means
  }
)

sinusoidal_embedding <- nn_module(
  initialize = function(embedding_min_frequency = 1, embedding_max_frequency = 1000, embedding_dim = 32) {
    self$frequencies <- nn_buffer(torch_exp(torch_linspace(
      start = torch_log(embedding_min_frequency),
      end = torch_log(embedding_max_frequency),
      steps = embedding_dim %/% 2
    )))
    self$angular_speeds <- nn_buffer(2*pi*self$frequencies)
  },
  forward = function(x) {
    embeddings <- self$angular_speeds*x
    out <- torch_cat(dim = 4, list(
      torch_sin(embeddings),
      torch_cos(embeddings)
    ))
    torch_transpose(out, 4, 2)
  }
)

diffusion <- nn_module(
  initialize = function(image_size, embedding_dim = 32, widths = c(32, 64, 96, 128), block_depth = 2) {
    self$unet <- unet(2*embedding_dim, embedding_dim, widths = widths, block_depth)
    self$embedding <- sinusoidal_embedding(embedding_dim = embedding_dim)

    self$conv <- nn_conv2d(image_size[1], embedding_dim, kernel_size = 1)
    self$emb_conv <- nn_conv2d(embedding_dim, embedding_dim, kernel_size = 1)

    self$upsample <- nn_upsample(size = image_size[2:3])

    self$activation <- swish()
    self$conv_out <- nn_conv2d(2*embedding_dim, image_size[1], kernel_size = 1)
    purrr::walk(self$conv_out$parameters, nn_init_zeros_)
  },
  forward = function(noisy_images, noise_variances) {
    embedded_variance <- noise_variances |>
      self$embedding() |>
      self$emb_conv() |>
      self$upsample()

    embedded_image <- noisy_images |>
      self$conv() |>
      self$activation()

    unet_input <- torch_cat(list(embedded_variance, embedded_image), dim = 2)
    unet_out <- unet_input |>
      self$unet()

    torch_cat(list(unet_out, embedded_variance), dim = 2) |>
      self$conv_out()
  }
)

diffusion_model <- nn_module(
  initialize = function(image_size, embedding_dim = 32, widths = c(32, 64, 96, 128), block_depth = 2,
                        signal_rate = c(0.02, 0.95)) {
    self$diffusion <- diffusion(image_size, embedding_dim, widths, block_depth)
    self$diffusion_schedule <- diffusion_schedule(signal_rate[1], signal_rate[2])
    self$image_size <- image_size
    self$normalize <- normalize(image_size[1])
  },
  denoise = function(images, rates) {
    pred_noises <- self$diffusion(images, rates$noise^2)
    pred_images <- (images - rates$noise * pred_noises) / rates$signal

    list(
      pred_noises = pred_noises,
      pred_images = pred_images
    )
  },
  forward = function(images, rates) {
    self$denoise(images, rates)
  },
  step = function() {
    images <- ctx$model$normalize(ctx$input)

    diffusion_times <- torch_rand(images$shape[1], 1, 1, 1, device = images$device)
    rates <- self$diffusion_schedule(diffusion_times)

    noises <- torch_clip(torch_randn_like(images), -1, 1)
    images <- rates$signal * images + rates$noise * noises

    c(pred_noises, pred_images) %<-% ctx$model(images, rates)
    loss <- nnf_l1_loss(noises, pred_noises)

    ctx$opt$zero_grad()
    loss$backward()
    ctx$opt$step()

    ctx$loss[[ctx$opt_name]] <- loss$detach()
  },
  reverse_diffusion = function(initial_noise, diffusion_steps) {
    step_size <- 1.0 / diffusion_steps

    next_noisy_images <- initial_noise

    diffusion_times <- torch_ones(c(initial_noise$shape[1], 1, 1, 1), device = ctx$accelerator$device)
    rates <- self$diffusion_schedule(diffusion_times)

    for (step in seq_len(diffusion_steps)) {
      noisy_images <- next_noisy_images

      c(pred_noises, pred_images) %<-% self$denoise(noisy_images, rates)

      # remix the predicted components using the next signal and noise rates
      diffusion_times <- diffusion_times - step_size
      rates <- self$diffusion_schedule(diffusion_times)

      next_noisy_images <- rates$signal * pred_images + rates$noise * pred_noises
    }

    pred_images |>
      self$normalize$denormalize() |>
      torch_clip(0, 1)
  },
  generate = function(num_images, diffusion_steps = 20) {
    device <- ctx$accelerator$device
    initial_noise <- torch_randn(c(num_images, self$image_size), device=device)
    self$reverse_diffusion(initial_noise)
  }
)

# model <- diffusion_model(c(3, 32, 32))
# diffusion_times <- torch_rand(32)
# rates <- diffusion_schedule(diffusion_times)
# x <- rates$noise
# a <- sinusoidal_embedding(x)
# nn_upsample(size = c(64,64))(a$view(c(a$shape, 1, 1)))

