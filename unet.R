box::use(torch[...])
box::use(purrr[map])

downsample <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv <- nn_conv2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = 3,
      stride = 2,
      padding = 1
    )
  },
  forward = function(input) {
    self$conv(input)
  }
)

swish <- nn_module(
  forward = function(input) {
    input*torch_sigmoid(input)
  }
)

resnet_block <- nn_module(
  initialize = function(in_channels, out_channels, kernel_size = 3) {
    self$conv_res <- nn_conv2d(
      in_channels,
      out_channels,
      kernel_size = 1,
      padding = "same"
    )
    self$batch_norm <- nn_batch_norm2d(num_features = in_channels)
    self$conv1 <- nn_conv2d(
      in_channels,
      out_channels,
      kernel_size = kernel_size,
      padding = "same"
    )
    self$activation <- swish()
    self$conv2 <- nn_conv2d(
      out_channels,
      out_channels,
      kernel_size = kernel_size,
      padding = "same"
    )
  },
  forward = function(input) {
    resid <- input |> self$conv_res()
    output <- input |>
      self$batch_norm() |>
      self$conv1() |>
      self$activation() |>
      self$conv2()
    resid + output
  }
)

down_block <- nn_module(
  initialize = function(in_channels, out_channels, block_depth = 2) {

    self$resnet <- nn_sequential(
      resnet_block(in_channels, out_channels),
      !!! map(seq_len(block_depth - 1), \(i) resnet_block(out_channels, out_channels))
    )

    self$downsample <- downsample(out_channels, out_channels)
  },
  forward = function(input) {
    input |>
      self$resnet() |>
      self$downsample()
  }
)

up_block <- nn_module(
  initialize = function(in_channels, out_channels, block_depth = 2) {
    self$upsample <- nn_upsample(scale_factor = 2)
    self$conv <- nn_conv2d(in_channels, out_channels, kernel_size = 1)
    self$resnet <- resnet_block(2*out_channels, out_channels)

    self$resnet <- nn_sequential(
      resnet_block(2*out_channels, out_channels),
      !!! map(seq_len(block_depth - 1), \(i) resnet_block(out_channels, out_channels))
    )
  },
  forward = function(x, skip) {
    x <- x |> self$upsample() |> self$conv()
    x <- torch_cat(list(x, skip), dim = 2)
    self$resnet(x)
  }
)

unet <- nn_module(
  initialize = function(in_channels, out_channels, widths = c(32, 64, 96, 128), block_depth = 2) {
    self$down_blocks <- nn_module_list()
    for (i in seq_along(widths)) {
      self$down_blocks$append(
        down_block(widths[i-1] %|% in_channels, widths[i], block_depth)
      )
    }

    self$up_blocks <- nn_module_list()
    widths <- rev(widths)
    for (i in seq_along(widths)) {
      self$up_blocks$append(
        up_block(widths[i], widths[i+1] %|% in_channels, block_depth)
      )
    }

    self$conv <- nn_conv2d(in_channels, out_channels, kernel_size = 1)
  },
  forward = function(x) {

    skips <- list(x)
    for (i in seq_along(self$down_blocks)) {
      skips[[i+1]] <- self$down_blocks[[i]](skips[[i]])
    }

    skips <- rev(skips)
    for (i in seq_along(self$up_blocks)) {
      skips[[i+1]] <- self$up_blocks[[i]](skips[[i]], skips[[i+1]])
    }

    self$conv(skips[[i+1]])
  }
)

`%|%` <- function (x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x))
    y
  else x
}

