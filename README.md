# TGAN
Generative adversarial training for synthesizing tabular data.  
This project was originally from [DAI-Lab @ MIT](https://DAI-Lab.github.io/TGAN)

# Overview
TGAN is a tabular data synthesizer. It can generate fully synthetic data from real data. Currently, TGAN can
generate numerical columns and categorical columns.

# Requirements

## Python

**TGAN** has been developed and runs on Python [3.5](https://www.python.org/downloads/release/python-356/),
[3.6](https://www.python.org/downloads/release/python-360/) and
[3.7](https://www.python.org/downloads/release/python-370/).

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system where **TGAN**
is run.

# Installation

The simplest and recommended way to install TGAN is using `pip`:

```
pip install tgan
```

# Data Format

## Input Format

In order to be able to sample new synthetic data, **TGAN** first needs to be *fitted* to
existing data. The input data for this *fitting* process has to be a single table that satisfies the following
rules:

* Has no missing values.
* Has columns of types `int`, `float`, `str` or `bool`.
* Each column contains data of only one type.

An example of such a tables would be:

| str_column | float_column | int_column | bool_column |
|------------|--------------|------------|-------------|
|    'green' |         0.15 |         10 |        True |
|     'blue' |         7.25 |         23 |       False |
|      'red' |        10.00 |          1 |       False |
|   'yellow' |         5.50 |         17 |        True |

As you can see, this table contains 4 columns: `str_column`, `float_column`, `int_column` and
`bool_column`, each one being an example of the supported value types. Notice aswell that there is
no missing values for any of the rows.

**NOTE**: It's important to have properly identifed which of the columns are numerical, which means
that they represent a magnitude, and which ones are categorical, as during the preprocessing of
the data, numerical and categorical columns will be processed differently.

## Output Format

The output of **TGAN** is a table of sampled data with the same columns as the input table and as
many rows as requested.

## Dataset
This project uses [Hepatitis C Virus (HCV) for Egyptian patients Data Set](https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+%28HCV%29+for+Egyptian+patients) that is being hosted by the UCI Machine Learning Repository.  
You can also download datasets that we have ran through cleaning.  
[Clean HCV data](https://drive.google.com/file/d/16y24a7n-IF8Lqq16lgfJj7K8Uy5Pq3TR/view?usp=sharing)  
[Discretized HCV data](https://drive.google.com/file/d/1XjjUoHOwD71hypDzLTtf0j4HOZPCjDVg/view?usp=sharing)

# Quickstart

```
python train.py
```

Before running the training script, make sure your original data path is placed in a correct location.  
Please refer to the section right below to modify parameters of the training model.  
If you would like to use pre-trained models, there are 3 different models.  
[Model 1](https://drive.google.com/file/d/1jxWPmmzmlWkJW2txgaN-93cB22GZK0Ob/view?usp=sharing)  
[Model 2](https://drive.google.com/file/d/1ef6ZUkWwrt9TxWV9Ze-Dz5NXFNXLVTQ6/view?usp=sharing)  
[Model 3](https://drive.google.com/file/d/1FpgRx_JKlP3yuiZLFfnyrBzLDjngJOeH/view?usp=sharing)  
[Model Descriptions](https://drive.google.com/file/d/1BzIwGYd9dnZeLEd0c-X3EZo1zrlp4bcL/view?usp=sharing)

# Model Parameters

If you want to change the default behavior of `TGANModel`, such as as different `batch_size` or
`num_epochs`, you can do so by passing different arguments when creating the instance.

## Model general behavior

* continous_columns (`list[int]`, required): List of columns indices to be considered continuous.
* output (`str`, default=`output`): Path to store the model and its artifacts.

## Neural network definition and fitting

* max_epoch (`int`, default=`100`): Number of epochs to use during training.
* steps_per_epoch (`int`, default=`10000`): Number of steps to run on each epoch.
* save_checkpoints(`bool`, default=True): Whether or not to store checkpoints of the model after each training epoch.
* restore_session(`bool`, default=True): Whether or not continue training from the last checkpoint.
* batch_size (`int`, default=`200`): Size of the batch to feed the model at each step.
* z_dim (`int`, default=`100`): Number of dimensions in the noise input for the generator.
* noise (`float`, default=`0.2`): Upper bound to the gaussian noise added to categorical columns.
* l2norm (`float`, default=`0.00001`): L2 reguralization coefficient when computing losses.
* learning_rate (`float`, default=`0.001`): Learning rate for the optimizer.
* num_gen_rnn (`int`, default=`400`): Number of units in rnn cell in generator.
* num_gen_feature (`int`, default=`100`): Number of units in fully connected layer in generator.
* num_dis_layers (`int`, default=`2`): Number of layers in discriminator.
* num_dis_hidden (`int`, default=`200`): Number of units per layer in discriminator.
* optimizer (`str`, default=`AdamOptimizer`): Name of the optimizer to use during `fit`, possible
  values are: [`GradientDescentOptimizer`, `AdamOptimizer`, `AdadeltaOptimizer`].

# Data Generation
```
python generate.py --model [Model Name] --num_samples [Number of samples to generate] --output_path [Path to save the resulting csv file]

e.g.) python generate.py --model model_x --num_samples 1400 --output_path examples/model_x.csv
```

Using aforementioned models, these are generated synthetic data  
[Synthetic Data with Model 1](https://drive.google.com/file/d/1BNld7dheIDWQ-YRIF9LAFQ391osmp0aN/view?usp=sharing)  
[Synthetic Data with Model 2](https://drive.google.com/file/d/1Uzd1ecmCxhB0okAzRDUdhIv9z4dSk6hn/view?usp=sharing)  
[Synthetic Data with Model 3](https://drive.google.com/file/d/1b0YBrQsonLPHwJJppvb3tJdWki--J1c7/view?usp=sharing)  

Please refer to [distribution notebook](distribution.ipynb) for how each model produces a different distribtuion.

# Research

The first **TAGN** version was built as the supporting software for the [Synthesizing Tabular Data using Generative Adversarial Networks](https://arxiv.org/pdf/1811.11264.pdf) paper by Lei Xu and Kalyan Veeramachaneni.

The exact version of software mentioned in the paper can be found in the releases section as the [research pre-release](https://github.com/DAI-Lab/TGAN/releases/tag/research)

# What's next?

For more details about **TGAN** and all its possibilities and features, please check the
[project documentation site](https://DAI-Lab.github.io/TGAN/)!

# Citing TGAN

If you use TGAN for yor research, please consider citing the following paper (https://arxiv.org/pdf/1811.11264.pdf):

If you use TGAN, please cite the following work:

> Lei Xu, Kalyan Veeramachaneni. 2018. Synthesizing Tabular Data using Generative Adversarial Networks.

```LaTeX
@article{xu2018synthesizing,
  title={Synthesizing Tabular Data using Generative Adversarial Networks},
  author={Xu, Lei and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:1811.11264},
  year={2018}
}
```
# Summary
This project was done for CMPE256 project @ SJSU on Fall 2019.
