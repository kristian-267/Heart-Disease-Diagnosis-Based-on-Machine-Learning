import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

filename = 'data/processed.cleveland.data'
df = pd.read_csv(filename, names=col_names)
df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric)
df.num = df.num.replace([2, 3, 4], 1)

raw_data = df.dropna().values

cols = range(0, 13)
# norm_cols = [0, 3, 4, 7]
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:, -1]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape
C = len(classNames)

print(df.isnull().sum())

# PCA
Y = X - np.ones((N, 1))*X.mean(axis=0)
Y = Y * (1/np.std(Y, 0))
U, S, Vh = svd(Y, full_matrices=False)
rho = (S*S) / (S*S).sum()
cum_rho = np.cumsum(rho)
threshold = 0.9

plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), cum_rho, 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.show()

V = Vh.T

Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

plt.plot(X[:, i], X[:, j], 'o')

f = plt.figure()
plt.title('NanoNose data')
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])

# Output result to screen
plt.show()

# Plot PCA of the data
f = plt.figure()
plt.title('NanoNose data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

pcs = [0, 1, 2, 3]
legendStrs = ['PC'+str(e+1) for e in pcs]
bw = .2
r = np.arange(1, M+1)

plt.figure()
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()

print('PC2:')
print(V[:, 1].T)

health_data = Y[y == 0, :]

print('health observation')
print(health_data[0, :])
print('...and its projection onto PC2')
print(health_data[0, :]@V[:, 1])

r = np.arange(1, X.shape[1]+1)
plt.bar(r, np.std(X, 0))
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Heart disease: attribute standard deviations')

Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2, 0))

Ys = [Y, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.4)
plt.title('NanoNose: Effect of standardization')
nrows = 3
ncols = 2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U, S, Vh = svd(Ys[k], full_matrices=False)
    V = Vh.T  # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k == 1: V = -V; U = -U;

    # Compute variance explained
    rho = (S * S) / (S * S).sum()

    # Compute the projection onto the principal components
    Z = U * S;

    # Plot projection
    plt.subplot(nrows, ncols, 1 + k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y == c, i], Z[y == c, j], '.', alpha=.5)
    plt.xlabel('PC' + str(i + 1))
    plt.xlabel('PC' + str(j + 1))
    plt.title(titles[k] + '\n' + 'Projection')
    plt.legend(classNames)
    plt.axis('equal')

    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols, 3 + k)
    for att in range(V.shape[1]):
        plt.arrow(0, 0, V[att, i], V[att, j])
        plt.text(V[att, i], V[att, j], attributeNames[att])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)),
             np.sin(np.arange(0, 2 * np.pi, 0.01)));
    plt.title(titles[k] + '\n' + 'Attribute coefficients')
    plt.axis('equal')

    # Plot cumulative variance explained
    plt.subplot(nrows, ncols, 5 + k);
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.title(titles[k] + '\n' + 'Variance explained')

plt.show()

plt.boxplot(X)
plt.xticks(range(1,14),attributeNames)
plt.ylabel('')
plt.title('boxplot')
plt.show()

plt.figure(figsize=(20, 20))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)

plt.show()

corr = pd.DataFrame(X).corr()
