#| requires:
#|  - file: data
#|    target-type: link
#| train:
#|  sourcecode:
#|    - '*.R'
#| output-scalars:
#|   - '(\key): (\value)'

box::use(luz[...])
box::use(./diffusion[diffusion_model])
box::use(./dataset[make_dataset])
box::use(torch[...])
box::use(./callbacks[callback_generate_samples])

tfevents::set_default_logdir(fs::path("logs", gsub("[:punct: -]", "", lubridate::now())))

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
dataset_name <- "pets"

image_size <- c(3 , 64, 64)

optimizer <- if (optimizer == "adam") {
  optim_adam
} else if (optimizer == "adamw") {
  torchopt::optim_adamw
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
  setup(optimizer = optimizer) %>%
  set_hparams(
    image_size = image_size,
    block_depth = block_depth,
    signal_rate = c(min_signal_rate, max_signal_rate),
    loss = loss
  ) %>%
  set_opt_hparams(lr = lr, weight_decay = weight_decay)

# finder <- luz::lr_finder(model, data = dataset)
# plot(finder) + ggplot2::coord_cartesian(ylim = c(0, 2))

fitted <- model %>%
  fit(
    dataset, epochs = epochs, dataloader_options = list(batch_size = batch_size, drop_last = TRUE), verbose = TRUE,
    callbacks = list(
      luz_callback_lr_scheduler(lr_step, step_size = patience/2, gamma = 0.1^(1/3)),
      luz_callback_early_stopping(monitor = "train_loss", min_delta = 0.0005, patience = patience),
      callback_generate_samples(num_images = 20, diffusion_steps = 20)
    )
  )

with_no_grad({
  x <- fitted$model$normalize$denormalize(fitted$model$generate(20, diffusion_steps = 20)$to(device = "mps"))
})

saveRDS(as_array(x), "generated.rds")
luz_save(fitted, path = "luz_model.luz")

to_cpu <- function(x) x$to(device="cpu")

for (i in 1:20) {
  x[i,..] %>%
    to_cpu() %>%
    torch_transpose(1,3) %>%
    torch_transpose(1,2) %>%
    torch_clip(0, 1) %>%
    as_array() %>%
    as.raster() %>%
    plot()
}

