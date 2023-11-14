# %% Import and standardize datas
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
df = df.drop('atemp', axis=1)

df.head()



from sklearn.preprocessing import StandardScaler

X_full = df.values
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
df = pd.DataFrame(X_full_scaled, columns=df.columns)

X = df.drop(columns=['cnt']).values
attributeNames = df.columns.drop(['cnt']).tolist()
N, M = X.shape
y = df['cnt'].values



# %% Define variables for double cross validation

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
from toolbox_02450 import train_neural_net, draw_neural_net
import torch
from toolbox_02450 import rlr_validate
from scipy import stats


## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 5
K2 = 5
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
size_val = np.empty(K2)
size_par = np.empty(K1)
size_test = np.empty(K1)
  
#ANN
h = 0
h_values = [1, 300, 375, 440]
h_number = len(h_values)
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 5000
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
Eval_ANN = np.empty((K2,h_number))
Egen_ANN_temp = np.empty((h_number))
Etest_ANN = np.empty(K1)
optimal_ANN = np.empty(K1) 


class Squeeze(torch.nn.Module):
    def forward(self, input):
        return torch.squeeze(input)

#baseline 
Error_val_nofeatures = np.empty((K2,1))
Etest_nofeatures = np.empty((K1,1))

#Linear regression
lambdas = np.power(10.,range(-1,3))
opt_lambda = np.empty((K1,1)) 
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))
w_rlr = np.empty((M,K1))
w = np.empty((M,K2,len(lambdas)))
Error_test_rlr = np.empty((K1,1))
Eval_rlr = np.empty((K2,len(lambdas))) 
Egen_rlr_temp = np.empty((len(lambdas)))

 # %% Double crosss validation
 
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
    
    X_test_ANN = torch.from_numpy(X_test).float()
    y_test_ANN = torch.from_numpy(y_test).float()
    X_par_ANN = torch.from_numpy(X_par).float()
    y_par_ANN = torch.from_numpy(y_par).float()
    
    CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
    
    k_2 = 0
    for train_index, val_index in CV2.split(X_par):
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_val = X[val_index,:]
        y_val = y[val_index]      
        size_val[k_2] = len(val_index) 
        X_train_rlr = X[train_index,:]
        X_val_rlr = X[val_index,:]
        
        X_train_ANN = torch.from_numpy(X_train).float()
        y_train_ANN = torch.from_numpy(y_train).float()
        X_val_ANN = torch.from_numpy(X_val).float()
        y_val_ANN = torch.from_numpy(y_val).float()
    
    
        #ANN
        
        
        h_index = 0
        for h in h_values:
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h), #M features to n_hidden_units
                torch.nn.Tanh(),   # 1st transfer function,
                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                Squeeze() # remove the extra dimension
                ) 
            
            
            #Train the model
            net, final_loss, learning_curve = train_neural_net(model,
                                                                loss_fn,
                                                                X=X_train_ANN,
                                                                y=y_train_ANN,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            # Test model
            y_test_est_ANN = net(X_val_ANN)
            
            # Determine errors and errors
            se = (y_test_est_ANN.float()-y_val_ANN.float())**2 # squared error
            Eval_ANN[k_2, h_index] = (sum(se).type(torch.float)/len(y_val)).data.numpy() #mean
            h_index = h_index + 1
            
            
            
            
        
        #Linear regression
        
        # Standardize the training and set set based on training set moments
        mu_rlr = np.mean(X_train[:, 1:], 0)
        sigma_rlr = np.std(X_train[:, 1:], 0)
        X_train_rlr[:, 1:] = (X_train[:, 1:] - mu_rlr) / sigma_rlr
        X_val_rlr[:, 1:] = (X_val[:, 1:] - mu_rlr) / sigma_rlr
        
        Xty = X_train_rlr.T @ y_train
        XtX = X_train_rlr.T @ X_train_rlr
        
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,k_2,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            Eval_rlr[k_2,l] = np.power(y_val-X_val_rlr @ w[:,k_2,l].T,2).mean(axis=0)
    
            
            
            
        k_2 = k_2 + 1
        
        
           
        
        
    







    #ANN    
    h_index = 0
    for h in h_values:  
        Egen_ANN_temp[h_index] = np.dot(size_val, Eval_ANN[:, h_index])/size_par[k_1]
        h_index = h_index + 1
    h_opti = h_values[int(np.argmin(Egen_ANN_temp))]  #choice of parameter
    optimal_ANN[k_1] = h_opti 
    
    # Define the model
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M,h_opti), #M features to n_hidden_units
            torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.Linear(h_opti, 1), # n_hidden_units to 1 output neuron
            Squeeze() # remove the extra dimension
            ) 
    #Train the model
    net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_par_ANN,
                                                        y=y_par_ANN,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
    #Test model
    y_test_est_ANN = net(X_test_ANN)
    # Determine errors and errors
    se = (y_test_est_ANN.float()-y_test_ANN.float())**2 # squared error
    Etest_ANN[k_1] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean 
    
    
    
    
    
    #Linear regression
    for l in range(0,len(lambdas)):
        Egen_rlr_temp[l] = np.dot(size_val, Eval_rlr[:, l])/size_par[k_1]
    opt_lambda[k_1] = lambdas[np.argmin(Egen_rlr_temp)]
    #we should obtain the same kfold than for the rest since the function used inside rlr_validate
    #is the same
    
    mu[k_1, :] = np.mean(X_par[:, 1:], 0)
    sigma[k_1, :] = np.std(X_par[:, 1:], 0)
    
    X_par_rlr[:, 1:] = (X_par[:, 1:] - mu[k_1, :] ) / sigma[k_1, :] 
    X_test_rlr[:, 1:] = (X_test[:, 1:] - mu[k_1, :] ) / sigma[k_1, :] 
    
    Xty = X_par_rlr.T @ y_par
    XtX = X_par_rlr.T @ X_par_rlr
    
    lambdaI = opt_lambda[k_1] * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k_1] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    Error_test_rlr[k_1] = np.square(y_test-X_test_rlr @ w_rlr[:,k_1]).sum(axis=0)/y_test.shape[0]
    
    
    #baseline
    Etest_nofeatures[k_1] = np.square(y_test-y_par.mean()).sum()/y_par.shape[0]
                  
    k_1 = k_1 + 1
    
# %%

#Final generalisation error baseline
Egen_nofeatures = np.dot(size_test,  Etest_nofeatures)/N     
#Final generalisation error ANN
Egen_ANN  = np.dot(size_test, Etest_ANN)/N
#Final generalisation error rlr
Egen_rlr = np.dot(size_test, Error_test_rlr)/N

#%% Tableau


from IPython.display import display, HTML

# Visualizza i risultati

# Crea un MultiIndex per le colonne
columns = pd.MultiIndex.from_tuples([
    ('Outer fold', 'i'),
    ('ANN', 'h<sub>i</sub>'),
    ('ANN', 'E<sub>i</sub><sup>test</sup>'),
    ('Linear Regression', '&lambda;<sub>i</sub>'),
    ('Linear Regression', 'E<sub>i</sub><sup>test</sup>'),
    ('Baseline', 'E<sub>i</sub><sup>test</sup>')
])
outer_fold_i = np.reshape(range(1, K1 + 1), (K1))
opt_lambda = np.reshape(opt_lambda, (K1))
Error_test_rlr = np.reshape(Error_test_rlr, (K1))
Etest_nofeatures = np.reshape(Etest_nofeatures, (K1))

# Convert lists to pandas Series before applying the format() function
dff = pd.DataFrame(list(zip(pd.Series(outer_fold_i), pd.Series(optimal_ANN), pd.Series(Etest_ANN), pd.Series(opt_lambda), pd.Series(Error_test_rlr), pd.Series(Etest_nofeatures))), columns=columns)

    
# Apply the format() function
df_styled = dff.style.set_properties(**{'text-align': 'center'}).format("{:.2f}")

# Aggiungi CSS personalizzato per allineare le intestazioni delle colonne
styles = """
    <style>
        th {
            text-align: center;
        }
    </style>
"""

# Visualizza il DataFrame come HTML
display(HTML(styles + df_styled.to_html()))

# %% Statistical comparison

import numpy as np, scipy.stats as st

K = 10
CV = model_selection.KFold(K,shuffle=True)

# store predictions.
yhat = []
y_true = []
i=0
for train_index, test_index in CV.split(X, y): 
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    X_train_rlr_stat = X[train_index,:]
    X_test_rlr_stat = X[test_index,:]
    X_train_ANN = torch.from_numpy(X_train).float()
    y_train_ANN = torch.from_numpy(y_train).float()
    X_test_ANN = torch.from_numpy(X_test).float()
    y_test_ANN = torch.from_numpy(y_test).float()
    

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    dy = []
    
    # linear regression 
    # trova lambda_star come il parametro di controllo ottimale
    index = np.argmin(Error_test_rlr)
    lambda_star = opt_lambda[index]
    mu_rlr = np.mean(X_train[:, 1:], 0)
    sigma_rlr = np.std(X_train[:, 1:], 0)
    X_train_rlr_stat[:, 1:] = (X_train[:, 1:] - mu_rlr) / sigma_rlr
    X_test_rlr_stat[:, 1:] = (X_test[:, 1:] - mu_rlr) / sigma_rlr
    
    Xty = X_train_rlr_stat.T @ y_train
    XtX = X_train_rlr_stat.T @ X_train_rlr_stat
    
    # Compute parameters for current value of lambda and current CV fold
    # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
    lambdaI = lambda_star * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    w_rlr_stat = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    y_est = X_test_rlr_stat @ w_rlr_stat
    dy.append( y_est )









    # ann
    h_star = 200
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, h_star), #M features to n_hidden_units
            torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.Linear(h_star, 1), # n_hidden_units to 1 output neuron
            Squeeze() # remove the extra dimension
            ) 
    # Train the model
    net, final_loss, learning_curve = train_neural_net(model,
                                                      loss_fn,
                                                      X=X_train_ANN,
                                                      y=y_train_ANN,
                                                      n_replicates=n_replicates,
                                                      max_iter=max_iter)
    # Test model
    y_est = net(X_test_ANN).detach().numpy()
    dy.append( y_est )









    # baseline
    y_est = y_train.mean()*np.ones(len(y_test)) 
    dy.append( y_est )


    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    i+=1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)


## Confidence interval test

# %%
# Significance level used for the statistical tests
alpha = 0.05

# First comparison
zA = np.abs(y_true - yhat[:,0] ) ** 2
zB = np.abs(y_true - yhat[:,1] ) ** 2

z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value


print()
print("Linear regresssion vs. ANN")
print("z = mean(Z_A-Z_B) estimator", z.mean(), " CI: ", CI, "p-value", p)
print()

# Second comparison
zA = np.abs(y_true - yhat[:,1] ) ** 2
zB = np.abs(y_true - yhat[:,2] ) ** 2

z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print()
print("ANN vs. Baseline")
print("z = mean(Z_A-Z_B) estimator", z.mean(), " CI: ", CI, "p-value", p)
print()

# Third comparison
zA = np.abs(y_true - yhat[:,0] ) ** 2
zB = np.abs(y_true - yhat[:,2] ) ** 2

z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print()
print("Linear  Regression vs. Baseline")
print("z = mean(Z_A-Z_B) estimator", z.mean(), " CI: ", CI, "p-value", p) 
print()

