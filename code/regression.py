import numpy as np
import pandas as pd
from sklearn import model_selection
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, show, grid, gca, subplots, xlim, ylim, plot, boxplot, xticks)
import torch
import scipy.stats as st

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def find_outliers(X):
    iqr = X.quantile(0.75) - X.quantile(0.25)
    up_cap = X.quantile(0.75) + 1.5 * iqr
    low_cap = X.quantile(0.25) - 1.5 * iqr
    return (X < low_cap) | (X > up_cap)


def train_neural_net(model, loss_fn, X, y, n_replicates=3, max_iter=1000, tolerance=1e-6):
    # Specify maximum number of iterations for training
    best_final_loss = 1e100
    best_net = model()
    best_learning_curve = []
    for r in range(n_replicates):
        # Make a new net (calling model() makes a new initialization of weights)
        net = model()

        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)

        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        # optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)

        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())

        # Train the network
        learning_curve = []  # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X)  # forward pass, predict labels on training set
            loss = loss_fn(y_est, y)  # determine loss
            loss_value = loss.data.numpy()  # get numpy array instead of tensor
            learning_curve.append(loss_value)  # record loss for later display

            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value - old_loss) / old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value

            # do backpropagation of loss and optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss_value < best_final_loss:
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve


def correlated_ttest(y_true, yhat, alpha=0.05):
    y_true = np.concatenate(y_true)
    yhat = np.concatenate(yhat)

    # note our usual setup I ttest only makes sense if m=1.
    zA = np.abs(y_true - yhat[:, 0]) ** 2
    zB = np.abs(y_true - yhat[:, 1]) ** 2
    z = zA - zB

    CI = np.round(st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)), 2)  # Confidence interval
    p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return p, CI


def one_level_crossvalidation(X, y, lambdas, h_units, K, ann=False):
    CV = model_selection.KFold(K, shuffle=True)
    M = X.shape[1]
    w = np.empty((M, K, len(lambdas)))
    train_error = np.empty((K, len(lambdas)))
    test_error = np.empty((K, len(lambdas)))
    train_err_nofeatures = np.empty(K)
    test_err_nofeatures = np.empty(K)
    f = 0
    y = y.squeeze()

    # Parameters for neural network classifier
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 1000

    errors = np.zeros((K, len(h_units)))  # make a list for storing generalizaition error in each loop

    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(axis=0)
            test_error[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)
        train_err_nofeatures[f] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        test_err_nofeatures[f] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

        if ann:
            # Extract training and test set for current CV fold, convert to tensors
            Xn_train = torch.Tensor(X_train[:, 1:])
            yn_train = torch.Tensor(y_train[:, np.newaxis])
            Xn_test = torch.Tensor(X_test[:, 1:])
            yn_test = torch.Tensor(y_test[:, np.newaxis])

            Mn = M - 1

            for hid in range(0, len(h_units)):
                # Define the model
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(Mn, h_units[hid]),  # M features to n_hidden_units
                    torch.nn.Tanh(),  # 1st transfer function,
                    torch.nn.Linear(h_units[hid], 1),  # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                )
                loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

                # Train the net on training data
                net, final_loss, learning_curve = train_neural_net(model,
                                                                   loss_fn,
                                                                   X=Xn_train,
                                                                   y=yn_train,
                                                                   n_replicates=n_replicates,
                                                                   max_iter=max_iter)

                # Determine estimated class labels for test set
                yn_test_est = net(Xn_test)

                # Determine errors and errors
                se = (yn_test_est.float() - yn_test.float()) ** 2  # squared error
                mse = (sum(se).type(torch.float) / len(yn_test)).data.numpy()  # mean
                errors[f, hid] = mse[0]  # store error rate for current CV fold

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))
    train_r2 = (train_err_nofeatures.sum() - train_error[:,
                                             np.where(lambdas == opt_lambda)].sum()) / train_err_nofeatures.sum()
    test_r2 = (test_err_nofeatures.sum() - test_error[:,
                                           np.where(lambdas == opt_lambda)].sum()) / test_err_nofeatures.sum()

    ann_errors = np.mean(errors, axis=0)
    opt_error = np.min(ann_errors)
    try:
        opt_h = h_units[np.where(ann_errors == opt_error)[0][0]]
    except:
        opt_h = 1

    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, train_r2, test_r2, opt_h, ann_errors


def two_level_crossvalidation(X, y, lambdas, h_units, K1, K2):
    CV = model_selection.KFold(K1, shuffle=True)
    M = X.shape[1]

    # Initialize variables
    Error_train = np.empty((K1, 1))
    Error_test = np.empty((K1, 1))
    Error_train_rlr = np.empty((K1, 1))
    Error_test_rlr = np.empty((K1, 1))
    Error_train_nofeatures = np.empty((K1, 1))
    Error_test_nofeatures = np.empty((K1, 1))
    w_rlr = np.empty((M, K1))
    mu = np.empty((K1, M - 1))
    sigma = np.empty((K1, M - 1))
    w_noreg = np.empty((M, K1))

    alpha = 0.05

    # Parameters for neural network classifier
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 1000

    k = 0
    Error_ANN = np.zeros((K1, 1))  # make a list for storing generalizaition error in each loop
    ANN_h = np.zeros((K1, 1))
    Reg_lamda = np.zeros((K1, 1))

    y_true = []
    yhatAB = []
    yhatAC = []
    yhatBC = []
    for train_index, test_index in CV.split(X, y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, train_r2, test_r2, opt_h, ann_errors = one_level_crossvalidation(
            X_train, y_train, lambdas, h_units, K2, ann=True)

        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        Error_test_nofeatures[k] = np.round(np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0], 2)

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
        Error_test_rlr[k] = np.round(np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0], 2)

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:, k] = np.linalg.lstsq(XtX, Xty, rcond=None)[0].squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
        Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]

        y_true.append(y_test)
        yhatB = X_test @ w_rlr[:, k]
        yhatB = yhatB[:, np.newaxis]
        yhatC = np.array([y_test.mean()] * len(y_test))[:, np.newaxis]

        Reg_lamda[k] = np.round(opt_lambda, 2)

        # Extract training and test set for current CV fold, convert to tensors
        Xn_train = torch.Tensor(X_train[:, 1:])
        yn_train = torch.Tensor(y_train[:, np.newaxis])
        Xn_test = torch.Tensor(X_test[:, 1:])
        yn_test = torch.Tensor(y_test[:, np.newaxis])

        Mn = M - 1

        # Define the model
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(Mn, opt_h),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(opt_h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=Xn_train,
                                                           y=yn_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)

        # Determine estimated class labels for test set
        yn_test_est = net(Xn_test)

        # Determine errors and errors
        se = (yn_test_est.float() - yn_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(yn_test)).data.numpy()  # mean
        Error_ANN[k] = np.round(mse, 2)  # store error rate for current CV fold
        ANN_h[k] = opt_h

        yhatA = yn_test_est.float().detach().numpy()

        yhatAB.append(np.concatenate([yhatA, yhatB], axis=1))
        yhatAC.append(np.concatenate([yhatA, yhatC], axis=1))
        yhatBC.append(np.concatenate([yhatB, yhatC], axis=1))

        k += 1

    # Initialize parameters and run test appropriate for setup I
    alpha = 0.05
    pAB, CIAB = correlated_ttest(y_true, yhatAB, alpha=alpha)
    pAC, CIAC = correlated_ttest(y_true, yhatAC, alpha=alpha)
    pBC, CIBC = correlated_ttest(y_true, yhatBC, alpha=alpha)

    comp = {
        'CIAB': CIAB,
        'CIAC': CIAC,
        'CIBC': CIBC,
        'pAB': pAB,
        'pAC': pAC,
        'pBC': pBC,
    }

    return Error_train, Error_test, Error_train_rlr, Error_test_rlr, Error_train_nofeatures, Error_test_nofeatures, Error_ANN, ANN_h, Reg_lamda, comp


col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
             'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

filename = 'data/processed.cleveland.data'
df = pd.read_csv(filename, names=col_names)
df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric)
df = df.dropna()

target = 'oldpeak'
cols = len(df.columns)
target_idx = col_names.index(target)

y = pd.DataFrame({target: df.iloc[:, target_idx]})
features_idx = list(range(0, target_idx)) + list(range(target_idx + 1, cols))
X = df.iloc[:, features_idx]

# Remove outliers
outlier_mask = (find_outliers(df.loc[:, 'trestbps'])) | (find_outliers(df.loc[:, 'chol'])) | (
    find_outliers(df.loc[:, 'thalach']))
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask]
y = y[valid_mask]

onehot_attr = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']
for key in onehot_attr:
    X = pd.concat([X, pd.get_dummies(X[key], prefix=key)], axis=1)
X.drop(onehot_attr, axis=1, inplace=True)

attributeNames = list(X.columns)

X = X.values
y = y.values.flatten()

N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

X = X.astype(np.float32)
y = y.astype(np.float32)

attributeNames = [u'Offset'] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = K2 = 10

# Values of lambda
lambdas = np.power(10., np.arange(-2, 7, 0.01))

n_hidden_units = np.array(range(1, 11))

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, train_r2, test_r2, opt_h, ann_errors = one_level_crossvalidation(
    X, y, lambdas, n_hidden_units, K1)

figure(1, figsize=(12, 8))
semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner
# plot, since there are many attributes
# legend(attributeNames[1:], loc='best')

figure(2, figsize=(12, 8))
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error', 'test error (generalization error)'])
grid()

show()

# Display results
print('******* Results after implementing one-level cross validation *******')
print('Regularized linear regression:')
print('- Training error: {0}'.format(train_err_vs_lambda[np.where(lambdas == opt_lambda)][0]))
print('- Test error:     {0}'.format(test_err_vs_lambda[np.where(lambdas == opt_lambda)][0]))
print('- R^2 train:     {0}'.format(train_r2))
print('- R^2 test:     {0}\n'.format(test_r2))
print('- optimal lambda: {0}\n'.format(np.round(opt_lambda, 2)))
print('- minimum error:  {0}\n'.format(np.round(opt_val_err, 4)))

print('Weights:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m],
                                 np.round(mean_w_vs_lambda[m, np.where(lambdas == opt_lambda)][0][0], 4)))

Error_train, Error_test, Error_train_rlr, Error_test_rlr, Error_train_nofeatures, Error_test_nofeatures, Error_ANN, ANN_h, Reg_lamda, comp = two_level_crossvalidation(
    X, y, lambdas, n_hidden_units, K1, K2)

# Display results
print('******* Results after implementing two-level cross validation *******')
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format(
    (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format(
    (Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

print('Optimal hidden layer nums of ANN for each outer cross-validation fold:\n', ANN_h)
print('\n')
print('Loss of ANN for each outer cross-validation fold:\n', Error_ANN)
print('\n')
print('Optimal lambda of Linear Regression for each outer cross-validation fold:\n', Reg_lamda)
print('\n')
print('Loss of Linear Regression for each outer cross-validation fold:\n', Error_test_rlr)
print('\n')
print('Loss of Baseline for each outer cross-validation fold:\n', Error_test_nofeatures)
print('\n')
print('t-test results for ANN, Linear Regression, baseline pair-comparison:\n', comp)
