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
  - Uses mlpconv layers in their architecture-basically just 1x1 convolutional layers that are really just mlps repeated over the whole spatial domain (WxH).
  - Uses global average pooling-which they note may be understood as a regularizer.




## My Projects
This section contains references to papers useful for specific projects I have worked on. 

