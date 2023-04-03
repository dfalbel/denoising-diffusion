#| train:
#|  description: Trains the denoising diffusion model
#| requires:
#|  - file: data
#|    target-type: link

box::use(luz[...])
box::use(./diffusion[diffusion_model, diffusion_schedule_config])
box::use(./dataset[make_dataset])
box::use(torch[...])
box::use(./callbacks[callback_generate_samples, image_loss, noise_loss, plot_tensors])
box::use(./kid[metric_kid])
box::use(zeallot[...])

set.seed(1)
torch_manual_seed(1)

# opt hyper parameters
optimizer <- "adamw"
lr <- 1e-3
weight_decay <- 1e-4

# architecture pars
block_depth <- 2
embedding_dim <- 32
unet_widths <- c(32, 64, 96, 128)
schedule_type <- "cosine"
min_signal_rate <- 0.02
max_signal_rate <- 0.95

# training pars
batch_size <- 64
epochs <- 50
loss <- "l1"
#| choices: [noise, image]
loss_on <- "noise"

# dataset pars
#| description: The dataset used for the experiment
#| choices: [flowers, pets, debug]
dataset_name <- "flowers"
image_size <- c(3, 64, 64)
num_workers <- 0

optimizer <- if (optimizer == "adam") {
  opt_hparams <- list(lr = lr)
  optim_adam
} else if (optimizer == "adamw") {
  opt_hparams <- list(lr = lr, weight_decay = weight_decay)
  torchopt::optim_adamw
} else {
  rlang::abort("Optimizer not currently supported.")
}

loss <- if (loss %in% c("l1", "mae")) {
  nnf_l1_loss
} else if (loss == "mse") {
  nnf_mse_loss
}

c(dataset, valid_dataset) %<-% make_dataset(dataset_name, image_size[-1])

model <- diffusion_model |>
  setup(
    optimizer = optimizer,
    metrics = luz_metric_set(
      metrics = list(image_loss(), noise_loss())
    )
  ) |>
  set_hparams(
    image_size = image_size,
    block_depth = block_depth,
    loss = loss,
    widths = unet_widths,
    embedding_dim = embedding_dim,
    schedule = diffusion_schedule_config(schedule_type, min_signal_rate, max_signal_rate),
    loss_on = loss_on
  ) |>
  set_opt_hparams(!!!opt_hparams)

finder <- luz::lr_finder(model, data = dataset)
plot(finder) + ggplot2::coord_cartesian(ylim = c(0, 2))

fitted <- model |>
  fit(
    dataset,
    epochs = epochs,
    dataloader_options = list(batch_size = batch_size, num_workers = num_workers),
    verbose = TRUE,
    callbacks = list(
      callback_generate_samples(num_images = 36, diffusion_steps = 20),
      luz_callback_tfevents(histograms = FALSE)
    )
  )

results <- fitted %>%
  evaluate(
    data = as_dataloader(
      valid_dataset,
      batch_size = 2*batch_size,
      shuffle = TRUE,
      num_workers = num_workers
    ),
    metrics = list(
      image_loss(),
      noise_loss(),
      metric_kid()
    )
  )

print(reults)

luz_save(fitted, path = "luz_model.luz")
