#| requires:
#|  - file: data
#|    target-type: link
#| output-scalars:
#|   - '(\key): (\value)'

box::use(luz[...])
box::use(./diffusion[diffusion_model])
box::use(./dataset[make_dataset])
box::use(torch[...])
box::use(./callbacks[callback_generate_samples, image_loss, plot_tensors])

logdir <- fs::path("logs", gsub("[:punct: -]", "", lubridate::now()))
tfevents::set_default_logdir(logdir)

block_depth <- 2
lr <- 1e-3
optimizer <- "adamw"
batch_size <- 64
epochs <- 1000
min_signal_rate <- 0.02
max_signal_rate <- 0.95
patience <- 200
weight_decay <- 1e-4
loss <- "l1"
embedding_dim <- 32
dataset_name <- "flowers"

image_size <- c(3, 64, 64)

optimizer <- if (optimizer == "adam") {
  optim_adam
} else if (optimizer == "adamw") {
  torch::optim_adamw
} else {
  rlang::abort("Optimizer not currently supported.")
}

loss <- if (loss %in% c("l1", "mae")) {
  nnf_l1_loss
} else if (loss == "mse") {
  nnf_mse_loss
}

dataset <- make_dataset(dataset_name, image_size[-1])

model <- diffusion_model %>%
  setup(optimizer = optimizer, metrics = list(image_loss())) %>%
  set_hparams(
    image_size = image_size,
    block_depth = block_depth,
    signal_rate = c(min_signal_rate, max_signal_rate),
    loss = loss,
    widths = c(32, 64, 96, 128),
    embedding_dim = embedding_dim
  ) %>%
  set_opt_hparams(lr = lr, weight_decay = weight_decay)

finder <- luz::lr_finder(model, data = dataset)
plot(finder) + ggplot2::coord_cartesian(ylim = c(0, 2))

fitted <- model %>%
  fit(
    dataset, epochs = epochs, dataloader_options = list(batch_size = batch_size), verbose = TRUE,
    callbacks = list(
      luz_callback_lr_scheduler(lr_step, step_size = patience/2, gamma = 0.1^(1/3)),
      luz_callback_early_stopping(monitor = "train_loss", min_delta = 0.0005, patience = patience),
      callback_generate_samples(num_images = 20, diffusion_steps = 20),
      luz_callback_tfevents(logdir = logdir, histograms = TRUE),
      luz_callback_gradient_clip()
    )
  )

luz_save(fitted, path = "luz_model.luz")
