import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def gradient_descent(x, y, lr=0.01, iters=2000, stop=1e-6):
    w, b, costs = 0.0, 0.0, []
    for i in range(iters):
        y_pred = w*x + b
        cost = np.mean((y - y_pred)**2)

        if costs and abs(costs[-1]-cost) <= stop:
            break
        costs.append(cost)

        dw = -(2/len(x))*np.sum(x*(y-y_pred))
        db = -(2/len(x))*np.sum(y-y_pred)
        w -= lr*dw
        b -= lr*db

        if i % 100 == 0:
            print(f"Iteration {i+1}: Cost {cost}, Weight {w}, Bias {b}")

    plt.plot(costs, 'r.')
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations"); plt.ylabel("Cost")
    plt.show()
    return w, b

# Data
X = np.array([32.5,53.4,61.5,47.4,59.8,55.1,52.2,39.2,48.1,52.5,45.4,
              54.3,44.1,58.1,56.7,48.9,44.6,60.2,45.6,38.8])
Y = np.array([31.7,68.7,62.5,71.5,87.2,78.2,79.6,59.1,75.3,71.3,55.1,
              82.4,62.0,75.3,81.4,60.7,82.8,97.3,48.8,56.8])

Xn = StandardScaler().fit_transform(X.reshape(-1,1)).flatten()
w, b = gradient_descent(Xn, Y)

print("Estimated Weight:", w, "Estimated Bias:", b)

Y_pred = w*Xn + b
plt.scatter(X, Y, marker='*', color='black', label='Data Points')
plt.plot(X, Y_pred, 'r--', label='Fitted Line')
plt.legend(); plt.show()
