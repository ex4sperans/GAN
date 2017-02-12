# GAN

This repo provides several implementations of various forms of Generative Advesarial Networks (GAN)

1. Original GAN (https://arxiv.org/abs/1406.2661)
2. Wasserstein GAN (https://arxiv.org/abs/1701.07875)

To run the experiments, execute ```train_gan.py```

To specify which model to train, use flag ```--model```. Say, to train GAN, use ```--model GAN```.
Additional parameters could be specified as well. 

To plot sample, execute ```plot.py``` with proper model flag.

