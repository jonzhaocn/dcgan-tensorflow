# dcgan-tensorflow

## Version
* tensorflow 1.4.0
* python 3.5

## Results
### loss
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/discrimination_loss.jpg" title="discriminator loss">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/discriminator_loss_fake.jpg" title="discriminator loss of fake image">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/discriminator_loss_true.jpg" title="discriminator loss of true image">
</p>
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/generator_loss.jpg" title="generator loss">
</p>

### generating images
the model converges rapidly, but it's easy to be model collapse
* iteration == 1000
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/999.jpg">
</p>

* iteration == 25000
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/24999.jpg">
</p>

* iteration == 50000
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/49999.jpg">
</p>

* iteration == 96000
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/95999.jpg">
</p>

* iteration == 97000
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/96999.jpg">
</p>

* iteration == 98000
<p align="center">
  <img src="https://github.com/JZhaoCH/dcgan-tensorflow/blob/master/images/97999.jpg">
</p>

## Refereces
1. `Radford A, Metz L, Chintala S. Unsupervised representation learning with deep convolutional generative adversarial networks[J]. arXiv preprint arXiv:1511.06434, 2015.`
2. `https://github.com/carpedm20/DCGAN-tensorflow`
3. `Liu Z, Luo P, Wang X, et al. Deep learning face attributes in the wild[C]//Proceedings of the IEEE International Conference on Computer Vision. 2015: 3730-3738.`
