box::use(./unet[unet, swish])
box::use(torch[...])
box::use(zeallot[...])

cosine_schedule <- nn_module(
  initialize = function(min_signal_rate = 0.02, max_signal_rate = 0.98) {
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

linear_schedule <- nn_module(
  initialize = function(min_signal_rate = 0.02, max_signal_rate = 0.98) {
    self$beta_start <- 0.0001
    self$beta_end <- 0.02
    self$scale <- 1000
    self$betas <- torch_linspace(self$beta_start, self$beta_end, steps = self$scale)
    self$alphas <- 1 - self$betas
    self$alpha_bars <- torch_cumprod(self = self$alphas, dim = 1)
    self$alpha_bars <- nn_buffer(torch_clip(self$alpha_bars, min_signal_rate, max_signal_rate))
  },
  forward = function(diffusion_times) {
    indexes <- 1L + as.integer((diffusion_times*(self$scale - 1))$cpu())
    alphas <- self$alpha_bars[indexes]
    list(
      signal = torch_sqrt(alphas)$view_as(diffusion_times),
      noise = torch_sqrt(1 - alphas)$view_as(diffusion_times)
    )
  }
)

diffusion_schedule <- nn_module(
  initialize = function(type = "cosine", min_signal_rate = 0.02, max_signal_rate = 0.98) {
    self$schedule <- if (type == "linear") {
      linear_schedule(min_signal_rate, max_signal_rate)
    } else if (type == "cosine") {
      cosine_schedule(min_signal_rate, max_signal_rate)
    } else {
      stop("unsupported")
    }
  },
  forward = function(diffusion_times) {
    self$schedule(diffusion_times)
  }
)

normalize <- nn_module(
  initialize = function(num_channels, num_steps = 55) {
    self$means <- nn_buffer(torch_zeros(1, num_channels, 1, 1))
    self$stds <- nn_buffer(torch_zeros(1, num_channels, 1, 1))
    self$step <- 0
    self$num_steps <- num_steps
    self$sample_size <- 0
  },
  forward = function(x) {

    if (self$step < self$num_steps) {
      self$step <- self$step + 1
      self$update_stats(x)
    }

    (x - self$means) / self$stds

    #x*2-1
  },
  update_stats = function(x) {
    means <- torch_mean(x, dim = c(1,3,4), keepdim = TRUE)
    vars <- torch_var(x, dim = c(1,3,4), keepdim = TRUE)

    sample_size <- x$size(1)
    total_size <- self$sample_size + sample_size

    w <- sample_size / total_size
    correction_factor <- sample_size * self$sample_size / (total_size^2) * ((self$means - means)^2)

    self$means$mul_(1-w)$add_(w*means)
    self$stds$pow_(2)$
      mul_(1-w)$
      add_(w*vars)$
      add_(correction_factor)$
      sqrt_()

    self$sample_size <- self$sample_size + sample_size
  },
  denormalize = function(x) {
    out <- x * self$stds + self$means
    torch_clip(out, 0, 1)
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
    out$permute(c(1,4,2,3))
  }
)

diffusion <- nn_module(
  initialize = function(image_size, embedding_dim = 32, widths = c(32, 64, 96, 128), block_depth = 2) {
    self$unet <- unet(2*embedding_dim, embedding_dim, widths = widths, block_depth)
    self$embedding <- sinusoidal_embedding(embedding_dim = embedding_dim)

    self$conv <- nn_conv2d(image_size[1], embedding_dim, kernel_size = 1)
    self$upsample <- nn_upsample(size = image_size[2:3])

    self$conv_out <- nn_conv2d(embedding_dim, image_size[1], kernel_size = 1)
    purrr::walk(self$conv_out$parameters, nn_init_zeros_)
  },
  forward = function(noisy_images, noise_variances) {
    embedded_variance <- noise_variances |>
      self$embedding() |>
      self$upsample()

    embedded_image <- noisy_images |>
      self$conv()

    unet_input <- torch_cat(list(embedded_image, embedded_variance), dim = 2)
    unet_output <- unet_input %>%
      self$unet() %>%
      self$conv_out()
  }
)

diffusion_model <- nn_module(
  initialize = function(image_size, embedding_dim = 32, widths = c(32, 64, 96, 128), block_depth = 2,
                        diffusion_schedule = cosine_schedule(0.02, 0.95), loss = NULL, loss_on = NULL) {
    self$diffusion <- diffusion(image_size, embedding_dim, widths, block_depth)
    self$diffusion_schedule <- diffusion_schedule
    self$image_size <- image_size
    self$normalize <- normalize(image_size[1])

    if (is.null(loss)) loss <- nnf_l1_loss
    self$loss <- loss
    self$loss_on <- if (is.null(loss_on)) "noise" else loss_on
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
    ctx$input <- images <- ctx$model$normalize(ctx$input)

    diffusion_times <- torch_rand(images$shape[1], 1, 1, 1, device = images$device)
    rates <- self$diffusion_schedule(diffusion_times)

    noises <- torch_randn_like(images)
    images <- rates$signal * images + rates$noise * noises

    ctx$pred <- ctx$model(images, rates)

    loss <- if (self$loss_on == "noise") {
      self$loss(noises, ctx$pred$pred_noises)
    } else if (self$loss_on == "image") {
      self$loss(images, ctx$pred$pred_images)
    }

    if (ctx$training) {
      ctx$opt$zero_grad()
      loss$backward()
      ctx$opt$step()
    }

    ctx$loss[[ctx$opt_name]] <- loss$detach()
  },
  reverse_diffusion = function(initial_noise, diffusion_steps) {
    noisy_images <- initial_noise

    diffusion_times <- torch_ones(c(initial_noise$shape[1], 1, 1, 1), device = initial_noise$device)
    rates <- self$diffusion_schedule(diffusion_times)

    # we want to combine with 'next' value in mind, thus we remove the first
    # value here
    for (step in seq(1, 0, length.out = diffusion_steps)[-1]) {
      c(pred_noises, pred_images) %<-% self$denoise(noisy_images, rates)

      # remix the predicted components using the next signal and noise rates
      diffusion_times <- torch_ones_like(diffusion_times)*step
      rates <- self$diffusion_schedule(diffusion_times)

      noisy_images <- rates$signal * pred_images + rates$noise * pred_noises
    }

    self$normalize$denormalize(pred_images)
  },
  generate = function(num_images, diffusion_steps = 20) {
    initial_noise <- torch_randn(c(num_images, self$image_size), device=self$device)
    self$reverse_diffusion(initial_noise, diffusion_steps = diffusion_steps)
  }
)
