import numpy as np
import pandas as pd
from sklearn import model_selection, tree
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, imshow, imread, box, axis, contour,
                              title, show, grid, gca, subplots, xlim, ylim, plot, boxplot, xticks, colorbar, subplot)
import torch
import scipy.stats as st
from itertools import chain
import sklearn.linear_model as lm
from platform import system
from os import getcwd

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from lib.toolbox_02450 import visualize_decision_boundary
import math


def find_outliers(X):
    iqr = X.quantile(0.75) - X.quantile(0.25)
    up_cap = X.quantile(0.75) + 1.5 * iqr
    low_cap = X.quantile(0.25) - 1.5 * iqr
    return (X < low_cap) | (X > up_cap)


def mcnemar(y_true, yhat, alpha=0.05):
    # perform McNemars test
    y_true = np.concatenate(y_true)
    yhat = np.concatenate(yhat)

    nn = np.zeros((2,2))
    c1 = yhat[:, 0] - y_true == 0
    c2 = yhat[:, 1] - y_true == 0

    nn[0, 0] = sum(c1 & c2)
    nn[0, 1] = sum(c1 & ~c2)
    nn[1, 0] = sum(~c1 & c2)
    nn[1, 1] = sum(~c1 & ~c2)

    n = sum(nn.flat)
    n12 = nn[0, 1]
    n21 = nn[1, 0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in st.beta.interval(1-alpha, a=p, b=q))

    p = 2*st.binom.cdf(min([n12, n21]), n=n12+n21, p=0.5)

    return thetahat, CI, p


def one_level_crossvalidation(X, Xm, y, lambdas, tc, K, ct=False):
    CV = model_selection.KFold(K, shuffle=True)
    y = y.squeeze()
    Mm = Xm.shape[1]

    # Initialize variable
    Error_test = np.empty((K, len(lambdas)))
    Error_train = np.empty((K, len(lambdas)))
    Tree_Error_test = np.empty((K, len(tc)))
    w = np.empty((Mm, K, len(lambdas)))
    offset = np.empty((K, len(lambdas)))

    k = 0
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        Xm_train = Xm[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        Xm_test = Xm[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(Xm_train, 0)
        sigma = np.std(Xm_train, 0)

        Xm_train = (Xm_train - mu) / sigma
        Xm_test = (Xm_test - mu) / sigma

        idx = np.argwhere(np.all(np.isnan(Xm_train[:, ]), axis=0))
        Xm_train = np.delete(Xm_train, idx, axis=1)
        Xm_test = np.delete(Xm_test, idx, axis=1)

        if ct:
            for i, t in enumerate(tc):
                # Fit decision tree classifier, Gini split criterion, different pruning levels
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
                dtc = dtc.fit(X_train, y_train.ravel())
                y_est_test = dtc.predict(X_test)
                # Evaluate misclassification rate over train/test data (in this CV fold)
                misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
                Tree_Error_test[k, i] = misclass_rate_test

        for i, l in enumerate(lambdas):
            # Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='auto',
                                        tol=1e-4, random_state=1,
                                        penalty='l2', C=1 / l, max_iter=1000)
            mdl.fit(Xm_train, y_train)
            y_test_est = mdl.predict(Xm_test)
            y_train_est = mdl.predict(Xm_train)
            coef = mdl.coef_.squeeze()
            if idx.size > 0:
                coef = np.concatenate((coef[:idx[0][0]], np.array(0), coef[idx[0][0]:]), axis=None)
            w[:, k, i] = coef
            offset[k, i] = mdl.intercept_[0]

            test_error_rate = np.sum(y_test_est != y_test) / len(y_test)
            train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
            Error_test[k, i] = test_error_rate
            Error_train[k, i] = train_error_rate

        k += 1

    opt_tc = tc[np.argmin(np.mean(Tree_Error_test, axis=0))]
    opt_val_err = np.min(np.mean(Error_test, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(Error_test, axis=0))]
    test_err_vs_lambda = np.mean(Error_test, axis=0)
    train_err_vs_lambda = np.mean(Error_train, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))
    mean_offset_vs_lambda = np.squeeze(np.mean(offset, axis=0))

    return opt_tc, opt_lambda, test_err_vs_lambda, train_err_vs_lambda, opt_val_err, mean_w_vs_lambda, mean_offset_vs_lambda


def two_level_crossvalidation(X, Xm, y, lambdas, tc, K1, K2):
    CV = model_selection.KFold(K1, shuffle=True)

    # Initialize variables
    Error_test = np.empty((K1, 1))
    Tree_Error_test = np.empty((K1, 1))
    Error_test_nofeatures = np.empty((K1, 1))

    Tree_tc = np.zeros((K1, 1))
    Multi_lamda = np.zeros((K1, 1))

    y_true = []
    yhatAB = []
    yhatAC = []
    yhatBC = []

    k = 0
    for train_index, test_index in CV.split(X, y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        Xm_train = Xm[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        Xm_test = Xm[test_index]
        y_test = y[test_index]

        opt_tc, opt_lambda, test_err_vs_lambda, train_err_vs_lambda, opt_val_err, mean_w_vs_lambda, mean_offset_vs_lambda = one_level_crossvalidation(X_train, Xm_train, y_train, lambdas, tc, K2, ct=True)

        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        # Standardize the training and set set based on training set moments
        mu = np.mean(Xm_train, 0)
        sigma = np.std(Xm_train, 0)

        Xm_train = (Xm_train - mu) / sigma
        Xm_test = (Xm_test - mu) / sigma

        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_tc)
        dtc = dtc.fit(X_train, y_train.ravel())
        y_est_test = dtc.predict(X_test)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        Tree_Error_test[k] = np.round(misclass_rate_test, 2)

        # Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
        mdl = lm.LogisticRegression(solver='lbfgs', multi_class='auto',
                                    tol=1e-4, random_state=1,
                                    penalty='l2', C=1 / opt_lambda, max_iter=1000)
        mdl.fit(Xm_train, y_train)
        y_test_est = mdl.predict(Xm_test)

        error_rate = np.sum(y_test_est != y_test) / len(y_test)
        Error_test[k] = np.round(error_rate, 2)

        Error_test_nofeatures[k] = np.round(np.sum(y_test != np.argmax(np.bincount(y_test))) / len(y_test), 2)

        y_true.append(y_test)
        yhatA = y_est_test[:, np.newaxis]
        yhatB = y_test_est[:, np.newaxis]
        yhatC = np.array([np.argmax(np.bincount(y_test))] * len(y_test))[:, np.newaxis]

        Tree_tc[k] = np.round(opt_tc, 2)
        Multi_lamda[k] = np.round(opt_lambda, 2)

        yhatAB.append(np.concatenate([yhatA, yhatB], axis=1))
        yhatAC.append(np.concatenate([yhatA, yhatC], axis=1))
        yhatBC.append(np.concatenate([yhatB, yhatC], axis=1))

        k += 1

    # Initialize parameters and run test appropriate for setup I
    alpha = 0.05
    thetahatAB, CIAB, pAB = mcnemar(y_true, yhatAB, alpha=alpha)
    thetahatAC, CIAC, pAC = mcnemar(y_true, yhatAC, alpha=alpha)
    thetahatBC, CIBC, pBC = mcnemar(y_true, yhatBC, alpha=alpha)

    comp = {
        'CIAB': CIAB,
        'CIAC': CIAC,
        'CIBC': CIBC,
        'pAB': pAB,
        'pAC': pAC,
        'pBC': pBC,
        'thetahatAB': thetahatAB,
        'thetahatAC': thetahatAC,
        'thetahatBC': thetahatBC,
    }

    return Tree_Error_test, Error_test, Error_test_nofeatures, Tree_tc, Multi_lamda, comp


col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
             'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

filename = 'data/processed.cleveland.data'
df = pd.read_csv(filename, names=col_names)
df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric)
df = df.dropna()

target = 'exang'
cols = len(df.columns)
target_idx = col_names.index(target)

y = pd.DataFrame({target: df.iloc[:, target_idx]})
features_idx = list(range(0, target_idx)) + list(range(target_idx + 1, cols))
X = df.iloc[:, features_idx]

# Remove outliers
outlier_mask = (find_outliers(df.loc[:, 'trestbps'])) | (find_outliers(df.loc[:, 'chol'])) | (
    find_outliers(df.loc[:, 'thalach'])) | (find_outliers(df.loc[:, 'oldpeak']))
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask]
y = y[valid_mask]

Xm = X

onehot_attr = ['num', 'cp', 'sex', 'restecg', 'fbs', 'slope', 'ca', 'thal']
for key in onehot_attr:
    Xm = pd.concat([Xm, pd.get_dummies(X[key], prefix=key)], axis=1)
Xm.drop(onehot_attr, axis=1, inplace=True)

attributeNames = list(X.columns)
attributeNamesm = list(Xm.columns)
classNames = np.unique(y[target])

X = X.values
Xm = Xm.values
y = y.values.flatten()
y = np.array(list(map(np.int32, y)))

X = X.astype(np.float32)
Xm = Xm.astype(np.float32)
y = y.astype(np.int32)

N, M = X.shape
Nm, Mm = Xm.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = K2 = 10

# Values of lambda
lambdas = np.power(10., np.arange(-5, 7, 0.01))

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 31, 1)

opt_tc, opt_lambda, test_err_vs_lambda, train_err_vs_lambda, opt_val_err, mean_w_vs_lambda, mean_offset_vs_lambda = one_level_crossvalidation(X, Xm, y, lambdas, tc, K1)

figure(1, figsize=(12, 8))
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Error rate (crossvalidation)')
legend(['Train error', 'test error (generalization error)'])
grid()
show()

print('******* Results after implementing one-level cross validation *******')
print('- optimal lambda: {0}\n'.format(np.round(opt_lambda, 2)))
print('- minimum error:  {0}\n'.format(np.round(opt_val_err, 4)))

w = mean_w_vs_lambda[:, np.where(lambdas == opt_lambda)].flatten()
offset = mean_offset_vs_lambda[np.where(lambdas == opt_lambda)].flatten()

print('Weights:')
print('offset     {:>15}'.format(np.round(offset[0], 4)))
for m in range(Mm):
    print('{:>15} {:>15}'.format(attributeNamesm[m], np.round(w[m], 4)))

Tree_Error_test, Error_test, Error_test_nofeatures, Tree_tc, Multi_lamda, comp = two_level_crossvalidation(
    X, Xm, y, lambdas, tc, K1, K2)

# Display results
print('******* Results after implementing two-level cross validation *******')
print('Optimal depth nums of Classification Tree for each outer cross-validation fold:\n', Tree_tc)
print('\n')
print('Loss of Classification Tree for each outer cross-validation fold:\n', Tree_Error_test)
print('\n')
print('Optimal lambda of Logistic Regression for each outer cross-validation fold:\n', Multi_lamda)
print('\n')
print('Loss of Logistic Regression for each outer cross-validation fold:\n', Error_test)
print('\n')
print('Loss of Baseline for each outer cross-validation fold:\n', Error_test_nofeatures)
print('\n')
print('McNemera’s test results for Classification Tree, Logistic Regression, baseline pair-comparison:\n', comp)
