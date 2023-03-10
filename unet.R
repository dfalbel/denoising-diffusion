box::use(torch[...])
box::use(purrr[map])
box::use(zeallot[...])

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
    self$conv1 <- nn_conv2d(
      in_channels,
      out_channels,
      kernel_size = kernel_size,
      padding = "same",
      bias = TRUE
    )
    self$batch_norm1 <- nn_batch_norm2d(num_features = in_channels)
    self$conv2 <- nn_conv2d(
      out_channels,
      out_channels,
      kernel_size = kernel_size,
      padding = "same",
      bias = TRUE
    )
    self$activation <- swish()
  },
  forward = function(input) {
    resid <- input |> self$conv_res()
    output <- input |>
      self$batch_norm1() |>
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
    skips <- list()
    for (i in seq_along(self$resnet_blocks)) {
      x <- self$resnet_blocks[[i]](x)
      skips[[i]] <- x
    }

    output <- self$downsample(x)
    list(
      output = output,
      skips = skips
    )
  }
)

up_block <- nn_module(
  initialize = function(in_channels, out_channels, block_depth = 2) {
    self$upsample <- nn_upsample(scale_factor = 2, mode = "bilinear")

    self$resnet_blocks <- nn_module_list(rlang::list2(
      resnet_block(2*in_channels, out_channels),
      !!! map(seq_len(block_depth - 1), \(i) resnet_block(out_channels + in_channels, out_channels))
    ))

    self$in_channels <- in_channels
    self$out_channels <- out_channels
    self$block_depth <- block_depth
  },
  forward = function(x, skips) {
    x <- x |> self$upsample()

    for (i in seq_len(length(self$resnet_blocks))) {
      x <- torch_cat(list(x, skips[[i]]), dim = 2)
      x <- self$resnet_blocks[[i]](x)
    }

    x
  }
)

unet <- nn_module(
  initialize = function(in_channels, out_channels, widths, block_depth) {
    self$down_blocks <- nn_module_list()
    for (i in seq_along(widths)) {
      self$down_blocks$append(
        down_block(widths[i-1] %|% in_channels, widths[i], block_depth)
      )
    }

    self$mid_blocks <- nn_module_list()
    self$mid_blocks$append(
      resnet_block(widths[length(widths)], 2*widths[length(widths)])
    )
    self$mid_blocks$append(
      resnet_block(2*widths[length(widths)], widths[length(widths)])
    )

    self$up_blocks <- nn_module_list()
    widths <- rev(widths)
    for (i in seq_along(widths)) {
      self$up_blocks$append(
        up_block(widths[i], widths[i+1] %|% out_channels, block_depth)
      )
    }
  },
  forward = function(x) {

    skips <- list()
    for (i in seq_along(self$down_blocks)) {
      c(x, skip) %<-% self$down_blocks[[i]](x)
      skips[[i]] <- skip
    }

    for (i in seq_along(self$mid_blocks)) {
      x <- self$mid_blocks[[i]](x)
    }

    skips <- rev(skips)
    for (i in seq_along(self$up_blocks)) {
      x <- self$up_blocks[[i]](x, rev(skips[[i]]))
    }

    x
  }
)

`%|%` <- function (x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x))
    y
  else x
}
