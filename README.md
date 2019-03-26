# Non-Local-Block-PyTorch

## 1 Self Attention / Non-local Block

In order to strengthen the original network because the conv operators only have local perception field, Self Attention is proposed.
The H and W of inputs bigger, the results better. So it works well in several final deconv layers.
usage: `from Self_Attn import Self_Attn_FM, Self_Attn_C`
</br>
#### 1.1 Feature map level attention

inputs: B * C * H * W feature maps
returns: out: self attention value + input feature maps; attention: B * N * N (N = H * W)
usage: firstly define the block by `net = Self_Attn_FN()`, and use it like `attn_out = net(conv_out)`
</br>
#### 1.2 Channel level attention

inputs: B * C * H * W feature maps
returns: out: self attention value + input feature maps; attention: B * c * c (c is the latent dimension)
usage: firstly define the block by `net = Self_Attn_C()`, then use it like `attn_out = net(conv_out)`
</br>

## 2 Spectral Norm Block

In order to keep the 1-Lipschiz condition, Spectral Norm is embedded to each conv layer.
inputs: B * C * H * W feature maps
returns: out: self attention value + input feature maps; attention: B * c * c (c is the latent dimension)
usage: firstly import this file `from Spectralnorm import SpectralNorm`, then use it like `out = SpectralNorm(conv(in))`
</br>

## 3 Reference
Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 7794-7803).
</br>
Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957.
