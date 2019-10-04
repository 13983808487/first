# 文件的读取，我们直接通过给定的`load_CIFAR10`模块读取数据。 
from load_data import load_CIFAR10   # 感谢这个magic函数，你不必要担心如何写读取的过程。如果想了解细节，可以参考此文件。
import numpy as np
import matplotlib.pyplot as plt

cifar10_dir = 'cifar-10-batches-py'  # 定义文件夹的路径：请不要修改此路径！ 不然提交后的模型不能够运行。

# 清空变量，防止重复导入多次。 
try:
   del X_train, y_train
   del X_test, y_test
   print('清除之前导入过的变量...Done!')
except:
   pass

# 读取文件，并把数据保存到训练集和测试集合。  
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# TODO 随机采样训练样本5000个和测试样本500个。训练样本从训练集里采样，测试样本从测试集里采样。
num_training = 5000
num_test = 500

np.random.seed(0)
X_train = X_train[[np.random.randint(50000, size=num_training)]]
y_train = y_train[[np.random.randint(50000, size=num_training)]]

X_test = X_test[[np.random.randint(10000, size=num_test)]]
y_test = y_test[[np.random.randint(10000, size=num_test)]]

print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train1 = np.reshape(X_train, (X_train.shape[0], -1))
X_test1 = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train1.shape, X_test1.shape) # 确保维度正确

# TODO 通过K折交叉验证构造最好的KNN模型，并输出最好的模型参数，以及测试集上的准确率。 
import time
start = time.time
# 训练数据： （X_train1, y_train）, 测试数据：(X_test1, y_test)
params_k = [1,3,5,7,9]  # 可以选择的K值
params_p = [1,2,3] # 可以选择的P值

parameters = {
    'n_neighbors' : params_k,
    'p' : params_p
}
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# 构建模型
model = GridSearchCV(KNeighborsClassifier(algorithm = 'kd_tree'),parameters, cv=5, n_jobs=-1, verbose=1 )
model.fit(X_train1, y_train)
# 输出最好的K和p值 
print(model.best_params_)

# 输出在测试集上的准确率
print(model.score(X_test1, y_test))
print(f'cost time: {time.time() - start}')