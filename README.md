Denoising Diffusion Implicit Models
================

This repository contains a torch/luz implementation of the Denoising
Diffusion Implicit Models. Code in this repository is heavily influenced
by code in Béres (2022) which is mostly based on (Song, Meng, and Ermon
2020) with a few ideas coming from (Nichol and Dhariwal 2021) and
(Karras et al. 2022).

The main code entry point is the `train.R` script used for start
training. The network architecture and training loop are defined in
`diffusion.R`.

## Sinusoidal embedding

A sinusoidal embedding is used to encode the diffusion times into the
model. The visualization below shows how diffusion times are mapped to
the embedding - assuming the dimension size of 32. Each row is a
embedding vector given the diffusion time. Sinusoidal embedding have
nice properties, like preserving the relative distances (Kazemnejad
2019).

![](README_files/figure-commonmark/sinusoidal-1.png)

## Sampling images

Images can be sampled from the model using the `generate` method.
Remember to always set the model into `eval()` mode before sampling, so
the batch normal layers are correctly applied.

``` r
box::use(torch[...])
box::use(./callbacks[plot_tensors])

fitted <- luz::luz_load("luz_model.luz")

with_no_grad({
  fitted$model$eval()
  x <- fitted$model$generate(36, diffusion_steps = 5)$to(device = "mps")
})

plot_tensors(x)
```

![](README_files/figure-commonmark/samples-1.png)

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-kerasDDIM" class="csl-entry">

Béres, András. 2022. “Denoising Diffusion Implicit Models.”
<https://keras.io/examples/generative/ddim/>.

</div>

<div id="ref-karras2022" class="csl-entry">

Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. 2022.
“Elucidating the Design Space of Diffusion-Based Generative Models.”
<https://doi.org/10.48550/ARXIV.2206.00364>.

</div>

<div id="ref-kazemnejad2019:pencoding" class="csl-entry">

Kazemnejad, Amirhossein. 2019. “Transformer Architecture: The Positional
Encoding.” *Kazemnejad.com*.
<https://kazemnejad.com/blog/transformer_architecture_positional_encoding/>.

</div>

<div id="ref-nichol2021" class="csl-entry">

Nichol, Alex, and Prafulla Dhariwal. 2021. “Improved Denoising Diffusion
Probabilistic Models.” <https://doi.org/10.48550/ARXIV.2102.09672>.

</div>

<div id="ref-song2020" class="csl-entry">

Song, Jiaming, Chenlin Meng, and Stefano Ermon. 2020. “Denoising
Diffusion Implicit Models.” <https://doi.org/10.48550/ARXIV.2010.02502>.

</div>

</div>
