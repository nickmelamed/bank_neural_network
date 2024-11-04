# Neural Network to Classify Bank Marketing Campaign Success

## Motivation and Overview

Neural Networks are one of the many supervised learning methods being deployed in ML. They are particularly effective as classification tools, where they can be given features and can predict an outcome based on said features. The Sequential Neural Network (SNN), which uses a stack of hidden layers to process features, is particularly effective with classification models. 

The goal of this project is to build a simple SNN, capable of classifying whether or not customers subscribed to a term deposit based on the results of a bank marketing campaign. 

## Data 

I will be using the Bank Marketing dataset from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). Note that I am specifically using the bank-additional-full.csv file for this analysis. 

## Analysis

`main.ipynb`: Contains all analysis for the project <br><br>
`hyperparam_tuning.py`: Contains functions used for training neural network using multiprocessing to speed up computation

## License 

This project is released under the terms of the MIT License.