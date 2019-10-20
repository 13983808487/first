import numpy as np
import pandas as pd

def gradient_descent():

    w = 0
    b = 0

    lr = 0.0000001
    num_iter = 100

    df = pd.read_csv('nyc-east-river-bicycle-counts.csv', index_col=0)
    df = df[df.columns[5:7]]
    x = df.values[:,0]
    y = df.values[:,1]

    for i in range(num_iter):
        y_hat = w*x + b
        w_g = -2*sum(x*(y - y_hat))/len(x)
        b_g = -2*sum((y - y_hat))/len(x)
        w -= lr * w_g
        b -= lr * b_g
        print(w, b)
    return w, b

if __name__ == "__main__":
    print(gradient_descent())
