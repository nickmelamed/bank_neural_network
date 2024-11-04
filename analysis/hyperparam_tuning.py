''' This .py file contains the functions for building and training a sequential neural network, picking hyperparameters with k-cross fold validation'''

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torch 
from torch import nn
from torch import optim 
from imblearn import over_sampling
from multiprocessing import Pool


# reading in the data here so we can run multiprocessing properly; explanation for cleaninig in main.ipynb 

df = pd.read_csv("../data/bank-additional-full.csv", delimiter = ";")
df = df[df.columns[~df.columns.isin(['emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed'])]]
df = df.drop(columns = ['default'])
df = df[(df['housing'] != 'unknown') & (df['loan'] != 'unknown')]
df_dummy = pd.get_dummies(df, columns = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome'], prefix_sep = ': ')
df_dummy['housing'] = np.where(df['housing'].values == 'yes', 1, 0)
df_dummy['loan'] = np.where(df['loan'].values == 'yes', 1, 0)
df_dummy['y'] = np.where(df['y'].values == 'yes', 1, 0)

# setting X and y for same reasons 

X = df_dummy[df_dummy.columns[~df_dummy.columns.isin(['y'])]]
y = df_dummy['y']

# functions for creating and training model 

def sequential_NN(multiplier, N_i = len(X.columns)):
    ''' Create a Sequental NN from PyTorch with two linear hidden layers and a sigmoid activation function
    
        Inputs:
            multiplier (float): what we multiply the number of input neurons by to get the number of neurons for the hidden layers
            N_i (int): The number of input neurons, which we take as the number of features in the model 
        
        Outputs:
            model (nn.Sequential): Sequential NN with each hidden layer having the calculated number of neurons
            
    '''
    
    N_h = int(N_i * multiplier) 
                                 
    model = nn.Sequential(
    nn.Linear(N_i, N_h),
    nn.ReLU(),
    nn.Linear(N_h, N_i),
    nn.ReLU(),
    nn.Linear(N_i, 1),
    nn.Sigmoid())
    
    return model 


def train_sgd(lr, model, X, y, num_epochs, batch_size, loss_fn = nn.BCELoss()):
    '''Trains a model through mini-batch Stochastic Gradient Descent (SGD), returns final BCE loss
    
        Inputs: 
            lr (float): learning rate for optimizer
            model (nn.Module): model to train 
            X (tensor): training data feature tensor
            y (tensor): training data outcome tensor 
            num_epochs (int): Number of times the entirety of the model is run through SGD 
            batch_size (int): Number of inputs processed in single iteration of SGD 
            loss_fn (nn.BCELoss()): Binary Cross-Entropy Loss function from PyTorch
        
    '''
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad() # resets the gradient for faster performance 
            loss.backward() # computes the gradient for the given batch 
            optimizer.step() # updates parameters based on gradient calculation 
            


def pred_prob(X, model):
    ''' Runs given data through our model, returning the predicted probabilities
    
        Inputs:
            X (tensor): tensor containing input data for the model
            model (nn.Module): model to run data on
        
        Outputs:
            probabilities (tensor): tensor containing the predicted probabilities for X 
        
    '''
    
    with torch.no_grad(): # we don't want to update our gradient; we just want to see the already classified data
        probabilities = model(X)
    return probabilities


def bce(predicted, actual):
    ''' Return BCE loss of our predicted probabilities
    
        Inputs:
            predicted (tensor): tensor containing predicted probabilities for X (output of pred_prob)
            actual (tensor): tensor containing true classifications 
            
        Outputs: 
            bce_loss (float): BCE loss of predicted probabilities
            
    '''
    
    loss_tensor = torch.mean(-torch.sum(actual * torch.log(predicted), 1))
    return loss_tensor.float()



def df_to_tensor(df, outcome):
    ''' Converts a df into a numpy array and then a Tensor with dtype float32 to be used in a PyTorch model 
        
        Inputs: 
            df (DataFrame): Input dataframe to be converted
            outcome (Boolean): Whether or not the df is an outcome vector; if it is, must be converted to 1D tensor for processing
        
        Outputs:
            as_tensor (Tensor): dtype float32 Tensor; use float32 as it is the input type for torch.nn neural networks
            
    '''
    
    df_np = df.to_numpy() # convert to numpy so we can use torch.from_numpy method
    if outcome: 
        return torch.from_numpy(df_np).reshape(-1, 1).to(torch.float32) # reshape makes the Tensor 1D if it is an outcome vector
    return torch.from_numpy(df_np).to(torch.float32)



def cross_val_loss(params):
    ''' Calculates cross validation loss across a k=5 fold, given the model parameters 
    
        Inputs: 
            params (tuple): Multiplier to attain number of neurons in hidden layers, learning rate, batch and epoch size for SGD 
            
        Outputs: 
            results (tuple): Cross validation loss along with parameters for multiprocessing 
    
    '''
    
    neurons, alpha, batch, epoch = params
    neural = sequential_NN(neurons)  # Create neural network with given hidden layer neurons
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Create a 5-fold cross-validator
    fold_losses = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]  # Split data into validation and training folds
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_tensor = df_to_tensor(X_train, outcome=False)
        y_train_tensor = df_to_tensor(y_train, outcome=True)

        train_sgd(alpha, neural, X_train_tensor, y_train_tensor, epoch, batch)  # Train our model w/ hyperparams

        X_val_tensor = df_to_tensor(X_val, outcome=False)
        y_val_tensor = df_to_tensor(y_val, outcome=True)

        probabilities = pred_prob(X_val_tensor, neural)
        fold_loss = bce(probabilities, y_val_tensor)
        fold_losses.append(fold_loss)  # Append all fold losses so we may take mean for cross-validation error

    return np.mean(fold_losses), neurons, alpha, batch, epoch




def best_hyperparameters(hidden_neurons, alphas, batches, epochs, X, y):
    '''Runs multiprocessing on cross_val_loss to determine combination of hyperparameters that minimize cross validation error
    
        Inputs:
        hidden_neurons (list): potential multipliers for input neurons to get hidden layer neurons
        alphas (list): potential learning rates for SGD
        batches (list): potential batch sizes for SGD
        epochs (list): potential epochs for SGD
        X (df): feature matrix 
        y (df): outcome vector 
        
        Outputs: 
        optimal_params (dict): neuron multiplier, learning rate, batch and epoch size that minimize cross validation error
        
    '''
    combinations = [(neurons, alpha, batch, epoch) 
                          for neurons in hidden_neurons 
                          for alpha in alphas 
                          for batch in batches 
                          for epoch in epochs]


    with Pool(processes=None) as pool:  # 'None' uses as many processes as there are CPU cores
        results = pool.map(cross_val_loss, combinations)

    min_cross_val_loss = float('inf') # ensures loss will be lesser and our for loop works 
    optimal_params = {'Hidden Neurons': 0, 'Learning Rate': 0, 'Batch Size': 0, 'Epochs': 0} # ensures our parameters are changed 

    for loss, neurons, alpha, batch, epoch in results:
        if loss < min_cross_val_loss:
            optimal_params['Hidden Neurons'] = neurons
            optimal_params['Learning Rate'] = alpha
            optimal_params['Batch Size'] = batch 
            optimal_params['Epochs'] = epoch

    return optimal_params


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    hidden_neurons = [2/3, (2 + 2/3)/2, 2]
    alphas = [0.0001, 0.001, 0.01]
    batches = [32, 64, 128]
    epochs = [50, 100, 200]
    
    optimal_params = best_hyperparameters(hidden_neurons, alphas, batches, epochs, X, y)
    print(optimal_params)