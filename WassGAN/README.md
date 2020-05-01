# Wasserstein Generative Adversarial Network

Saturating sigmoids proved to be a real problem in the Deep Convolutional GAN given in the ConvGAN/ directory. Even with high generator boost factors, the discriminator would often completely dominate. Because of that, I'm building a Wassersteing GAN in this directory.

The general motivation here is that with a vanilla GAN the discriminator is attempting to give a probability that a given image is real. If the discriminator is very good at knowing when the generator is giving it a fake image, it will give probabilities very close to zero. With sigmoid activation this will have a very small gradient, so if the discriminator gets smarter than the generator, it will be hard for the generator to catch up.

In a Wassersteing GAN, the discriminator is replaced with a critic that scores images on their realness, real images are given positive scores and fake images are given negative scores.

$$Loss=(1/n)\sum_{i=1}^n{y_i p_i}$$
$y_i=1$ if real
$y_i=-1$ if fake
$p_i$ is the critic score

the critic wants labels and scores to have the same sign

## Problems:
- the function is unbounded, the critic is incentivized to pick high p for true images and highly negative p for fake images. This can create giant gradients for the generator

## Solutions:
- clip weights to be in +- 0.01
- train critic for 5 epochs, then train the generator for 1 epoch