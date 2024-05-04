import numpy as np

X = 2 * np.random.rand(24, 1)
Y = 4 + 3 * X + np.random.randn(24, 1)
X_b = np.c_[np.ones((24, 1)), X]

theta = np.random.randn(2,1)

for _ in range(100):
    import pdb; pdb.set_trace()
    gradients = 2 / 24 * X_b.T.dot(X_b.dot(theta) - Y)
    theta = theta - 0.1 * gradients

print(theta)

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print(theta_best)
