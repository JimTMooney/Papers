# Papers
A repository to keep track of papers/articles I have read with their corresponding summaries. This is useful as a sanity check for remembering what each paper does for future citations. The papers are organized both by general category as well as specific projects I have worked on.

## General Categories
This section contains generic categories of papers. For initializing my in depth understanding of this field, I lifted several hundred of the first papers-as well as the initial organization of this repo-from https://github.com/maziarraissi/Applied-Deep-Learning (what a fantastic resource by the way for anyone interested in the field of deep learning).

### Optimization
* An overview of gradient descent algorithms [[link]](https://ruder.io/optimizing-gradient-descent/)
  - Batch/stochastic/mini-batch stochastic for different dataset sizes
  - Momentum/Nesterov to solve for ill-conditioning for cost in parameter-space
  - Adagrad/rmsprop/adadelta to solve for sparse parameter updates
  - Adam to solve for both simultaneously

### Regularization/Augmentation/Initialization

### Image Classification
* Multi-column Deep Neural Networks for Image Classification [[link]](https://arxiv.org/pdf/1202.2745.pdf)
  - Early DNN. Used on some of the smaller datasets for character recognition tasks.
  - Use convolutional layers/maxpooling/tanh activation. 
  - Multiple DNN's are trained concurrently similar to traditional ensemble methods. They call these multiple columns.
* ImageNet Classification with Deep Convolutional Neural Networks [[link]](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  - AlexNet used on ILSVRC. Massive improvement over previous results
  - Used ReLU, local response normalization, two GPU's with independent upper layer calculations, overlapping pooling, dropout. Outputs to fixed size latent.
* Dropout: A Simple Way to Prevent Neural Networks from Overfitting [[link]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  - Regularizer used by applying a binary (although sometimes stochastic-but not in this paper) mask over the parameter vector. Analagous to bagging
* Network In Network [[link]](https://arxiv.org/pdf/1312.4400.pdf)
  - Uses mlpconv layers in their architecture-basically just 1x1 convolutional layers that are really just mlps repeated over the whole spatial domain (WxH). Can be used to decrease the required number of parameters.
  - Uses global average pooling-which they note may be understood as a regularizer.
* Very Deep Convolutional Networks for Large-Scale Image Recognition [[link]](https://arxiv.org/pdf/1409.1556.pdf)
  - VGGNet -> More depth led to improved results. 
  - Repeated 3x3 convolutions instead of larger receptive fields (as in previous architectures). Developed notion of repeated blocks of same structures.
* Going deeper with convolutions [[link]](https://arxiv.org/pdf/1409.4842.pdf)
  - Inception block -> GoogLeNet -> Each block looks at multiple scales of an image and concatenates each of those views.
  - Uses multiple softmax layers. Can be understood as representing multiple scales.
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [[link]](https://arxiv.org/pdf/1502.03167.pdf)
  - Normalizing internal layers ensures that changing parameters do not impact the distribution of inputs to later layers (internal covariate shift)
  - Use learnable parameters for mean and variance of internal states when applying normalization.
  - Can be seen as regularizer given that each instance is not deterministic in outputs-batch of input images/soundswaves/etc. are tied together
* Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification [[link]](https://arxiv.org/pdf/1502.01852.pdf)
  - Introduces PReLU activation function to learn negative side of ReLU function
  - He initialization -> init weights with ReLU/PReLU in mind -> allows for much deeper models to converge better
* Maxout Networks [[link]](https://arxiv.org/pdf/1302.4389.pdf)
  - Activation function that incorporates principles from dropout
  - Use k mlps per each hidden unit, take only the maximum
* Regularization of Neural Networks using DropConnect [[link]](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf)
  - Generalization of dropout that randomly drops out the weights of a network.
* Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition [[link]](https://arxiv.org/pdf/1406.4729.pdf)
  - SPPNet -> can produce fixed-length vector regardless of input image size
  - After convolutional layers, use average pooling at different scales (1x1-global, 2x2, 3x3). Concatenate the resulting vectors from each scale and within each scale as vectors to feed into the fully connected layers
* Rethinking the Inception Architecture for Computer Vision [[link]](https://arxiv.org/pdf/1512.00567.pdf)
  - More performant minded inception architecture that will improve on efficiency.
  - 1 -> reduce bottlenecks, 2 -> use higher dimensions, 3 -> use 1x1s for reductions, 4 -> balance width and depth
  - Factorize convolutions into smaller convolutions with same size receptive field
  - Regularizes model by using label smoothing.
* Improving neural networks by preventing co-adaptation of feature detectors [[link]](https://arxiv.org/pdf/1207.0580.pdf)
  - Original dropout paper

## My Projects
This section contains references to papers useful for specific projects I have worked on. 

