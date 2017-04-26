# GAN

This repo provides several implementations of various forms of Generative Advesarial Networks (GAN)

1. Original GAN (https://arxiv.org/abs/1406.2661)
2. Wasserstein GAN (https://arxiv.org/abs/1701.07875)
3. Deep Convolutional GAN (https://arxiv.org/pdf/1511.06434.pdf)

At the moment, all the models are tested on MNIST dataset. 

To run the experiments, execute ```train_gan.py```

To specify which model to train, use flag ```--model```. Say, to train GAN, use ```--model GAN```.
Additional parameters could be specified as well. See train_gan.py for details.

To plot sample, execute ```plot.py``` with proper model flag.

Default parameters should (e.g. learning rate, batch size, etc.) work fine, but it's a good idea to try different settings.