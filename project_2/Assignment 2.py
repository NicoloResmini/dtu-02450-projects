import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show, savefig)
import numpy as np





df = pd.read_csv('hour.csv')

# PREPROCESSING

# Removing useless attributes
df = df.drop('dteday', axis=1)
df = df.drop('instant', axis=1)
df = df.drop('yr', axis=1)

# Applying sqrt to "cnt" (to make it a continuous variable)
df['cnt'] = np.sqrt(df['cnt'])

# Removing deprecated attributes after the sqrt transformation (cnt = casual + registered)
df = df.drop('casual', axis=1)
df = df.drop('registered', axis=1)

df.head()

# Format data like exercises
X = df.drop(columns=['cnt']).values
N, M = X.shape
y = df['cnt'].values
attributeNames = df.columns.drop('cnt').tolist()

# apply a feature transformation to your data matrix x such that each column has mean 0 and standard deviation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
df = pd.DataFrame(X, columns=attributeNames)







from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
from toolbox_02450 import train_neural_net, draw_neural_net
import torch
from toolbox_02450 import rlr_validate



## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 10
K2 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
size_val = np.empty((K2,1))
size_par = np.empty((K1,1))
size_test = np.empty((K1,1))

#ANN
h = 0
n_hidden_units_max = 5      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
Eval_ANN = np.empty((K2,n_hidden_units_max))
Egen_ANN_temp = np.empty((n_hidden_units_max, 1))
Etest_ANN = np.empty((K1,1))
optimal_ANN = np.empty((K1,1))  

#baseline 
Error_val_nofeatures = np.empty((K2,1))
Etest_nofeatures = np.empty((K1,1))

#Linear regression
lambdas = np.power(10.,range(-1,9))
opt_lambda = np.empty((K1,1)) 
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))
w_rlr = np.empty((M,K1))
Error_test_rlr = np.empty((K1,1))

# Initialize variables
#Features = np.zeros((M,K))
#Error_train = np.empty((K,1))
#Error_test = np.empty((K,1))
#Error_train_fs = np.empty((K,1))
#Error_test_fs = np.empty((K,1))
#Error_train_nofeatures = np.empty((K,1))
#Error_test_nofeatures = np.empty((K,1))


k_1 = 0
for par_index, test_index in CV1.split(X):
    
    
    X_par = X[par_index,:]
    y_par = y[par_index]
    X_test = X[test_index,:]
    y_test = y[test_index]      
    size_par[k_1] = len(par_index)
    size_test[k_1] = len(test_index)
    X_par_rlr = X[par_index,:]
    X_test_rlr = X[test_index,:]
    
    CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
    
    k_2 = 0
    for train_index, val_index in CV2.split(X_par):
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_val = X[val_index,:]
        y_val = y[val_index]      
        size_val[k_2] = len(val_index) 
        
        

    
        #Baseline
        Error_val_nofeatures[k_2] = np.square(y_val-y_train.mean()).sum()/y_train.shape[0]
    
        #ANN
        for h in (range(n_hidden_units_max) + np.ones(n_hidden_units_max, int)  ):
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                )
            
            
            # Convert numpy arrays to PyTorch Tensors
            X_train = torch.tensor(X_train, dtype=torch.float)
            y_train = torch.tensor(y_train, dtype=torch.float)
            X_val = torch.tensor(X_val, dtype=torch.float)
            y_val = torch.tensor(y_val, dtype=torch.float)

            # Train the model
            net, final_loss, learning_curve = train_neural_net(model,
                                                            loss_fn,
                                                            X=X_train,
                                                            y=y_train,
                                                            n_replicates=n_replicates,
                                                            max_iter=max_iter)

            # Test model
            y_test_est_ANN = net(X_val).squeeze()  # remove extra dimension

           
            # Determine errors and errors
            se = (y_test_est_ANN.float()-y_val.float())**2 # squared error
            Eval_ANN[k_2, h-1] = (sum(se).type(torch.float)/len(y_val)).data.numpy() #mean
            
        #Linear regression
            
            
        k_2 = k_2 + 1
        
        
           
        
        
    
        

    
    #baseline
    Egen_temp_nofeatures = np.dot(size_val, Error_val_nofeatures)/size_par[k_1]
    Etest_nofeatures[k_1] = np.square(y_test-y_par.mean()).sum()/y_par.shape[0]






    #ANN    
    for h in (range(n_hidden_units_max)) : 
        Egen_ANN_temp[h] = np.dot(size_val, Eval_ANN[:, h])/size_par[k_1]
    
    optimal_ANN[k_1] = np.argmin(Egen_ANN_temp) + 1  #choice of parameter
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, optimal_ANN[k_1]), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(optimal_ANN[k_1], 1), # n_hidden_units to 1 output neuron
                        )
    #Train the model
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_par,
                                                       y=y_par,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    #Test model
    y_test_est_ANN = net(X_test)
    # Determine errors and errors
    se = (y_test_est_ANN.float()-y_test.float())**2 # squared error
    Etest_ANN[k_1] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean 
    
    
    
    
    #Linear regression
    opt_val_err, opt_lambda[k_1], mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_par,y_par,lambdas,K2)
    #we should obtain the same kfold than for the rest since the function used inside rlr_validate
    #is the same
    
    mu[k_1, :] = np.mean(X_par[:, 1:], 0)
    sigma[k_1, :] = np.std(X_par[:, 1:], 0)
    
    X_par_rlr[:, 1:] = (X_par[:, 1:] - mu[k_1, :] ) / sigma[k_1, :] 
    X_test_rlr[:, 1:] = (X_test[:, 1:] - mu[k_1, :] ) / sigma[k_1, :] 
    
    Xty = X_par_rlr.T @ y_train
    XtX = X_par_rlr.T @ X_par_rlr
    
    lambdaI = opt_lambda[k_1] * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k_1] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    Error_test_rlr[k_1] = np.square(y_test-X_test_rlr @ w_rlr[:,k_1]).sum(axis=0)/y_test.shape[0]

                  
    k_1 = k_1 + 1
    
    
#Final generalisation error baseline
Egen_nofeatures = np.dot(size_test,  Etest_nofeatures)/N     
#Final generalisation error ANN
Egen_ANN  = np.dot(size_test, Etest_ANN)/N




    # Compute squared error with all features selected (no feature selection)
    #m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
   # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

