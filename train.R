#| requires:
#|  - file: pets
#|    target-type: link
#| sourcecode:
#|  - "*.R"
#| output-scalars:
#|   - '(\key): (\value)'

box::use(luz[...])
box::use(./diffusion[diffusion_model])
box::use(./dataset[train_ds])
box::use(torch[...])

block_depth <- 2
lr <- 1e-4
optimizer <- "adam"
batch_size <- 128
epochs <- 200

image_size <- c(3 , 64, 64)

optimizer <- if (optimizer == "adam") {
  optim_adam
} else {
  rlang::abort("Optimizer not currently supported.")
}

model <- diffusion_model %>%
  setup(optimizer = optimizer) %>%
  set_hparams(image_size = image_size, block_depth = block_depth) %>%
  set_opt_hparams(lr = lr)

fitted <- model %>%
  fit(
    train_ds, epochs = epochs, dataloader_options = list(batch_size = batch_size), verbose = TRUE,
    callbacks = list(
      luz_callback_lr_scheduler(lr_step, step_size = 50, gamma = 0.1)
    )
  )

with_no_grad({
  x <- fitted$model$generate(20, diffusion_steps = 100)$to(device = "cpu")
})

saveRDS(as_array(x), "generated.rds")
luz_save(fitted, path = "luz_model.luz")

for (i in 1:20) {
  x[i,..] %>%
    torch_transpose(1,3) %>%
    torch_transpose(1,2) %>%
    torch_clip(0, 1) %>%
    as_array() %>%
    as.raster() %>%
    plot()
}

