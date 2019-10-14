import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def beijing(n):
    df = pd.read_csv('beijing_house_price.csv')
    df = df.drop_duplicates()
    df.drop(df.columns[[6, 11]], axis=1,inplace=True)

    pearson = np.abs(df.corr(method='pearson').iloc[-1])
    X = pearson.sort_values(ascending=False)[1:4]
    features_names = X.index.values
    features = df[features_names]

    poly_features = PolynomialFeatures(n)
    features = poly_features.fit_transform(features)
    target = df.iloc[:, [-1]]
    
    X_train, X_test,y_train,  y_test = train_test_split(
            features, target, test_size=0.3, random_state=10)
    #X_train = poly_features.fit_transform(X_train)
    #X_test = poly_features.fit_transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)

    mea = mean_absolute_error(y_test, model.predict(X_test))

    return mea

if __name__ == '__main__':
    print(beijing(3))
