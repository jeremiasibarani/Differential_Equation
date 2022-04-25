import numpy as np
import matplotlib.pyplot as plt

"""
    an exmaple :
    
    y' = 2x + y
    y(0) = 1
    Solution :
    y = -2(x + 1) + 3e^x
    

"""

def f(x, y):
    return np.round(2*x + y, 6)

def eulerMethod(Y, X, h, i):
    return np.round(Y[i - 1] + h * f(X[i-1], Y[i-1]), 6)

def improvedEulerMethod(Y, h, X, i):
    return np.round(Y[i - 1] + h * (f(X[i - 1], Y[i - 1]) + f(X[i], eulerMethod(Y, X, h, i)))/2, 6)

def exactSolution(x):
    return np.round(-2 * (x + 1) + 3 * np.exp(x), 6)

def rungeKutta(X, Y, i, h):
    k1 = h * f(X[i - 1], Y[i - 1])
    k2 = h * f(X[i - 1] + h/2, Y[i - 1] + k1/2)
    k3 = h * f(X[i - 1] + h/2, Y[i - 1] + k2/2)
    k4 = h * f(X[i - 1] + h, Y[i - 1] + k3)
    K = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return Y[i-1] + K


def assignX(h, x0, n):
    X = [None] * n
    X[0] = x0
    for i in range(1, n):
        X[i] = np.round(X[i-1] + h, 6)
    return X

def display(ImprovedEuler, ExactSol, Euler, X, n):
    print("X \t\t Exact Solution \t\t\t Improved Euler \t\t\t Euler")
    for i in range(n):
        print("{} \t\t {} \t\t\t {} \t\t\t {}".format(X[i], ExactSol[i], ImprovedEuler[i], Euler[i]))


h = 0.2
n = 100
x0 = 0.0
y0 = 1

X = assignX(h, x0, n)
ImprovedEuler = [None] * n
ImprovedEuler[0] = y0

Euler = [None] * n
Euler[0] = y0

RungeKutta = [None] * n
RungeKutta[0] = y0

for i in range(1, n):
    ImprovedEuler[i] = improvedEulerMethod(ImprovedEuler, h, X, i)
    Euler[i] = eulerMethod(Euler, X, h, i)
    RungeKutta[i] = rungeKutta(X, RungeKutta, i, h)



ExactSolution = [exactSolution(item) for item in X]



LineWidth = 2
plt.plot(X, ImprovedEuler, label="Improved Euler Method", linewidth = LineWidth, color='red')
plt.scatter(X, ExactSolution, label="Exact Solution")
plt.plot(X, RungeKutta, label="Runge-Kutta Methode", linewidth = LineWidth, color='orange')
plt.plot(X, Euler, label="Euler Method", linewidth = LineWidth, color='green')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Graphical Way")
plt.grid()
plt.legend()
plt.show()


