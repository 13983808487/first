import pandas as pd
import numpy as np

def caculate_w():
    data = pd.read_csv('nyc-east-river-bicycle-counts.csv', index_col=0)
    x = data['Brooklyn Bridge'].values.reshape(-1,1)
    x = np.matrix(np.concatenate((x, np.ones_like(x)), axis=1))
    y = np.matrix(data['Manhattan Bridge'].values.reshape(-1,1))
    
    W = (x.T * x).I * x.T * y
    w = round(float(W[0]), 2)
    b = round(float(W[1]), 2)
    
    return w, b

if __name__ == "__main__":
    print(caculate_w())
