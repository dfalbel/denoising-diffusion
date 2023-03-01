box::use(torch[...])
box::use(purrr[map])

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
    self$batch_norm <- nn_batch_norm2d(num_features = in_channels, affine = FALSE)
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

    self$resnet_blocks <- nn_module_list(rlang::list2(
      resnet_block(in_channels, out_channels),
      !!! map(seq_len(block_depth - 1), \(i) resnet_block(out_channels, out_channels))
    ))

    self$downsample <- nn_avg_pool2d(kernel_size = c(2,2))
  },
  forward = function(x) {
    skips <- list(x)
    for (block in seq_along(self$resnet_blocks)) {
      skips[[block+1]] <- self$resnet_blocks[[block]](skips[[block]])
    }

    output <- self$downsample(skips[[length(skips)]])
    list(
      output = output,
      skips = skips[-1]
    )
  }
)

up_block <- nn_module(
  initialize = function(in_channels, out_channels, block_depth = 2) {
    self$upsample <- nn_upsample(scale_factor = 2, mode = "bilinear")

    self$resnet_blocks <- nn_module_list(rlang::list2(
      !!! map(seq_len(block_depth), \(i) resnet_block(2*in_channels, in_channels))
    ))

    self$conv_out <- nn_conv2d(in_channels, out_channels, kernel_size=1, padding="same")
  },
  forward = function(x, skips) {
    x <- x |> self$upsample()

    for (block in seq_len(length(self$resnet_blocks))) {
      x <- torch_cat(list(x, skips[[block]]), dim = 2)
      x <- self$resnet_blocks[[block]](x)
    }

    self$conv_out(x)
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

    self$activation <- swish()
    self$conv <- nn_conv2d(in_channels, out_channels, kernel_size = 1)
  },
  forward = function(x) {

    skips <- list(list(output = x))
    for (i in seq_along(self$down_blocks)) {
      skips[[i+1]] <- self$down_blocks[[i]](skips[[i]]$output)
    }

    skips <- rev(skips)
    output <- skips[[1]]$output
    for (i in seq_along(self$up_blocks)) {
      output <- self$up_blocks[[i]](output, rev(skips[[i]]$skips))
    }

    output |>
      self$activation() |>
      self$conv()
  }
)

`%|%` <- function (x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x))
    y
  else x
}

