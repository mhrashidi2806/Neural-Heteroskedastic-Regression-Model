# Neural-Heteroskedastic-Regression-Model
## Overview
This project implements a neural heteroskedastic regression model, addressing cases where target variability depends on input data. It combines a normal probability density model with neural networks, using experiments.py for training, and nn_base.py and nn_mods.py for model design. Modify these files to experiment with different architectures.
## Implemention 

For this problem, we develop a probabilistic regression model by combining the normal probability density model with neural network models. Specifically, we define  as a sigmoid neural network that predicts the mean of  given .

## Code Structure

### experiments.py: 
Contains the experiment code used to train and evaluate the model.

### nn_base.py: 
Implements the base neural network components required for the regression model.

### nn_mods.py: 
Contains modifications and additional neural network architectures used in the model.
