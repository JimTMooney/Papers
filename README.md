# Papers
A repository to keep track of papers/articles I have read with their corresponding summaries. This is useful as a sanity check for remembering what each paper does for future citations. The papers are organized both by general category as well as specific projects I have worked on.

## General Categories
This section contains generic categories of papers. For initializing my in depth understanding of this field, I lifted several hundred of the first papers-as well as the initial organization of this repo-from https://github.com/maziarraissi/Applied-Deep-Learning (which is a fantastic resource for anyone interested in the field of deep learning).

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
* Training Very Deep Networks [[link]](https://arxiv.org/pdf/1507.06228.pdf)
  - Highway networks -> kind of a precursor to resnets -> they borrow on LSTMS and use a transfer gate to choose how much input to pass thruogh from the previous layer.
* FITNETS: HINTS FOR THIN DEEP NETS [[link]](https://arxiv.org/pdf/1412.6550.pdf)
  - Builds on knowledge distillation by training student networks to have the same internal activations as the techer network (the student networks also use thinner width) -> use similary between states using parameterized transform if hidden state sizes differ
* Distilling the Knowledge in a Neural Network [[link]](https://arxiv.org/pdf/1503.02531.pdf)
  - Knowledge distillation -> train so that cross entropy between post-softmax distributions of trained teacher and current student is minimized (along with actual objective
* Curriculum Learning [[link]](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
  - Use easier samples first, then gradually increase to more difficult examples in the course of training -> borrows on concept of shaping from psychology
  - Can be considered similar in nature to continuation methods
  - Shapes Libraries! I have been looking for these. Some of the experiments in this paper use BasicShapes and GeomShapes -> follow up on this for project purposes
* Deeply-Supervised Nets [[link]](https://arxiv.org/pdf/1409.5185.pdf)
  - Each layer of the network attempts to predict the output at the same time as the final layer. This way the gradient can better be nearer to an actual prediction regardless of layer.
* Deep Residual Learning for Image Recognition [[link]](https://arxiv.org/pdf/1512.03385.pdf)
  - ResNets -> Use residual connections to better allow information to propagate forward and backward. Allow for information highways.
  - I like to think of this through the lens of communication -> inputs and outputs have to talk and longer paths only make the stuff communicated further from the intention
* Identity Mappings in Deep Residual Networks [[link]](https://arxiv.org/pdf/1603.05027.pdf)
  - Gathered tests on performance of various residual blocks. Found that pre-activation works best as it simply allows gradient to fully flow backward without scaling or loss of information. 
  - Tested various gating procedures, dropout, constant scaling as well as choosing the location of BN, ReLU, etc. 
* Densely Connected Convolutional Networks [[link]](https://arxiv.org/pdf/1608.06993.pdf)
  - DenseNet -> Use denseblocks - in each denseblock all previous layers connect with later layers.
* Deep Networks with Stochastic Depth [[link]](https://arxiv.org/pdf/1603.09382.pdf)
  - Use ResNet, but randomly drop layers -> later layers are dropped with a higher probability. Leads to networks which train faster but are still appropriately expressive
* Aggregated Residual Transformations for Deep Neural Networks [[link]](https://arxiv.org/pdf/1611.05431.pdf)
  - ResNeXt -> Combine ideas from inception and resnet -> Each residual block itself has multiple paths that are split up.
* Residual Networks Behave Like Ensembles of Relatively Shallow Networks [[link]](https://arxiv.org/pdf/1605.06431.pdf)
  - ResNets act much more like ensembles than deep networks -> ablation of some layers does not significantly degrade performance suggesting that layers are somewhat independent of each other
  - Most of the gradient goes along shorter paths -> longer paths do not contribute much to gradients. The gradient still vanishes over longer paths, but the introduction of shorter paths means the gradient can still reach earlier layers.

## My Projects
This section contains references to papers useful for specific projects I have worked on. 

