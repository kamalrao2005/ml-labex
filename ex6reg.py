import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
np.random.seed(42)
n = 100
t = np.arange(n).reshape(-1, 1)


tlin = 0.5 * t.flatten() + 5
nli = np.random.normal(0, 0.5, n)
ylin = tlin + nli

tnlin = 0.0001 * t.flatten()**5 - 0.02 * t.flatten()**3 + 0.5 * t.flatten()
season = 2 * np.sin(0.2 * t.flatten())
nnli = np.random.normal(0, 0.5, n)
ynonlin = tnlin + season + nnli


def scalexy(X, y):
    sx = StandardScaler()
    sy = StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1, 1)).ravel()
    return Xs, ys, sx, sy


def plotfit(X, y, yp, title):
    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, label="Data", alpha=0.7)
    plt.plot(X, yp, 'r', label="SVR", linewidth=2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


Xs, ys, _, sy = scalexy(t, ylin)
svrlin = SVR(kernel='linear', C=1, epsilon=0.1)
svrlin.fit(Xs, ys)
yp = sy.inverse_transform(svrlin.predict(Xs).reshape(-1, 1)).ravel()
plotfit(t, ylin, yp, "5.1 SVR on Linear Trend")


Xs, ys, _, sy = scalexy(t, ynonlin)
svrnonlin = SVR(kernel='linear', C=1, epsilon=0.1)
svrnonlin.fit(Xs, ys)
yp = sy.inverse_transform(svrnonlin.predict(Xs).reshape(-1, 1)).ravel()
plotfit(t, ynonlin, yp, "5.2 SVR on Nonlinear Trend")


plt.figure(figsize=(8, 4))
Cs = [0.001, 1, 10000]
Xs, ys, _, sy = scalexy(t, ylin)
for c in Cs:
    svr = SVR(kernel='linear', C=c, epsilon=0.1)
    svr.fit(Xs, ys)
    yp = sy.inverse_transform(svr.predict(Xs).reshape(-1, 1)).ravel()
    plt.plot(t, yp, label=f"C={c}")
plt.scatter(t, ylin, alpha=0.3)
plt.title("5.3 Effect of C (eps=0.1)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 4))
epsvals = [0.05, 0.1, 0.5]
Xs, ys, _, sy = scalexy(t, ylin)
for e in epsvals:
    svr = SVR(kernel='linear', C=1, epsilon=e)
    svr.fit(Xs, ys)
    yp = sy.inverse_transform(svr.predict(Xs).reshape(-1, 1)).ravel()
    plt.plot(t, yp, label=f"eps={e}")
plt.scatter(t, ylin, alpha=0.3)
plt.title("5.4 Effect of epsilon (C=1)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

poly = PolynomialFeatures(degree=2, include_bias=False)
tpoly = poly.fit_transform(t)
Xs, ys, _, sy = scalexy(tpoly, ylin)
svrpoly = SVR(kernel='linear', C=1, epsilon=0.1)
svrpoly.fit(Xs, ys)
yp = sy.inverse_transform(svrpoly.predict(Xs).reshape(-1, 1)).ravel()
plotfit(t, ylin, yp, "5.5 Poly Features (deg=2) + Linear SVR")


def rbffeats(X, gamma=0.1):
    d2 = np.sum((X[:, np.newaxis] - X[np.newaxis, :])**2, axis=2)
    return np.exp(-gamma * d2)

Phi = rbffeats(t, gamma=0.01)
Xs, ys, _, sy = scalexy(Phi, ylin)
svrrbf = SVR(kernel='linear', C=1, epsilon=0.1)
svrrbf.fit(Xs, ys)
Phit = rbffeats(t, gamma=0.01)
Phits = StandardScaler().fit_transform(Phit)
yp = sy.inverse_transform(svrrbf.predict(Phits).reshape(-1, 1)).ravel()
plotfit(t, ylin, yp, "5.6 RBF Feats + Linear SVR")


plt.figure(figsize=(8, 4))

# Poly
tpoly = poly.fit_transform(t)
Xp, yps, _, sp = scalexy(tpoly, ylin)
svrpoly.fit(Xp, yps)
predpoly = sp.inverse_transform(svrpoly.predict(Xp).reshape(-1, 1)).ravel()

# RBF
Phi = rbffeats(t, gamma=0.01)
Xr, yrs, _, sr = scalexy(Phi, ylin)
svrrbf.fit(Xr, yrs)
Phis = StandardScaler().fit_transform(Phi)
predrbf = sr.inverse_transform(svrrbf.predict(Phis).reshape(-1, 1)).ravel()

plt.plot(t, predpoly, label="Poly deg=2")
plt.plot(t, predrbf, label="RBF Feats")
plt.scatter(t, ylin, alpha=0.3)
plt.title("5.7 Poly vs RBF")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
