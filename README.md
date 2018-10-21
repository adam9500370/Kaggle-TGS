# Kaggle-TGS
PyTorch implementation for [Kaggle TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)

**15th / 3291** - Team with [atom1231](https://www.kaggle.com/atom1231)

I only share code for my best model in this repository.

The scores of my best 10-fold model by training from scratch: 0.887 on private LB / 0.872 on public LB.


## Model Design

A small U-shape [1] model (~8MB) with
* Pyramid pooling module, (dilated) residual blocks and auxiliary training [2]
* Spatial and channel squeeze & excitation module [3]
* Guided upsampling module [4]


## Training
* Input size: 129x129 with reflect padding
* Batch size: 40
* Augmentation
	- Random horizontally flip
	- Random translate with reflect padding
	- Random crop and resize
* Optimizer: Adam
* Weight decay: 1e-4
* Initial learning rate: 1e-3
* Total loss function
	- 1.00 Cross-entropy loss
	- 1.00 Lovász-softmax loss [5]
	- 0.05 Topology-aware loss [6]
* 10-fold CV (random split)

## Progress
* U-shape network with pyramid pooling module, (dilated) residual blocks and auxiliary training by cross-entropy and Lovász-softmax loss
	- Step LR decay by 0.5 every 50 epochs for first 200 epochs
	- 37s/epoch on single 1080TI
	- 0.870 on private LB / 0.852 on public LB
* +Topology-aware loss
	- Cosine annealing LR for 400 epochs (8 cycles)
	- 50s/epoch on single 1080TI
	- 0.877 on private LB / 0.859 on public LB
* +Spatial and channel squeeze & excitation module
	- Cosine annealing LR for 1400 epochs (14 cycles)
	- 54s/epoch on single 1080TI
	- 0.886 on private LB / 0.871 on public LB
* +Guided upsampling module
	- Cosine annealing LR for 200 epochs (1 cycle)
	- 76s/epoch on single 1080TI
	- 0.887 on private LB / 0.872 on public LB


## Ensemble
* Mean of the probability map from all 10-fold models (TTA with horizontally flip)
* Threshold of salt probaility: 0.5


## Reference
[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation (U-Net)](https://arxiv.org/abs/1505.04597)

[2] [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/abs/1612.01105)

[3] [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks (scSE)](https://arxiv.org/abs/1803.02579)

[4] [Guided Upsampling Network for Real-Time Semantic Segmentation (GUN)](https://arxiv.org/abs/1807.07466)

[5] [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)

[6] [Beyond the Pixel-Wise Loss for Topology-Aware Delineation](https://arxiv.org/abs/1712.02190)

[7] Part of code adapted from [meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

[8] Official implementation for [Lovász-Softmax loss](https://github.com/bermanmaxim/LovaszSoftmax)
