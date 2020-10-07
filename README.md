# MUTAG graph2vec Neural Classifier

This is a comparison of an anime recommender system using matrix factorization against a neural recommender system.

We're using the MUTAG dataset. The data can be found [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

A write up of our findings and methodology can be found at https://paul-tqh-nguyen.github.io/mutag_graph2vec_neural_classifier/ .

Here's a brief summary of our model's architecture:
* We embed the chemical compound structure via [graph2vec](https://arxiv.org/abs/1707.05005).
* We then put the embeddings through a deep neural network of dense layers with ReLU activation functions. 
* The result from the dense layers are passed through a fully-connected layer and a sigmoid to get the labels.

For more details, see our [write-up](https://paul-tqh-nguyen.github.io/mutag_graph2vec_neural_classifier/). 

### Tools Used

The following tools were utilized:

- [Optuna](https://optuna.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [PyTorch](https://pytorch.org/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)
- [Pandas](https://pandas.pydata.org/)
- [Pandarallel](https://github.com/nalepae/pandarallel)
- [Matplotllib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

Other Python libraries used include [contextlib](https://docs.python.org/3/library/contextlib.html), [typing-extensions](https://pypi.org/project/typing-extensions/), [abc](https://docs.python.org/3/library/abc.html), [more-itertools](https://more-itertools.readthedocs.io/en/stable/), [multiprocessing](https://docs.python.org/3/library/multiprocessing.html), [typing](https://docs.python.org/3/library/typing.html), [collections](https://docs.python.org/3/library/collections.html), [argparse](https://docs.python.org/3/library/argparse.html), [json](https://docs.python.org/3/library/json.html), [os](https://docs.python.org/3/library/os.html), and [logging](https://docs.python.org/3/library/logging.html).