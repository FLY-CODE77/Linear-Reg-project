import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from module import basic_node as nodes
from module import affine_MSE
from tqdm import trange
import multiprocessing

plt.style.use('seaborn')
np.random.seed(0)



def get_data_batch(dataset, batch_idx, batch_size, n_batch):
    if batch_idx is n_batch -1:
        batch = dataset[batch_idx*batch_size:]
    else:
        batch = dataset[batch_idx*batch_size : (batch_idx+1)*batch_size]
    return batch

### Start Dataset Preparation ###
n_sample = 200
h_order = 3

x_data1 = np.linspace(0.05, 1 - 0.05, n_sample).reshape(-1, 1)
y_data = np.sin(2*np.pi*x_data1) + 0.2*np.random.normal(0, 1, size = (n_sample,1))

x_data = np.zeros(shape = (n_sample, 1))
for order in range(1, h_order + 1):
    order_data = np.power(x_data1, order)
    x_data = np.hstack((x_data, order_data))

data = np.hstack((x_data, y_data))


batch_size = 32
n_batch = np.ceil(data.shape[0]/batch_size).astype(int)
feature_dim = x_data.shape[1]-1
Th = np.ones(shape = (feature_dim + 1,), dtype = np.float).reshape(-1, 1)

affine = affine_MSE.Affine_Function(feature_dim, Th)
cost = affine_MSE.MSE_Cost()

epochs, lr = 100000, 0.01
th_accum = Th.reshape(-1, 1)
cost_list = []

for epoch in trange(epochs):
    np.random.shuffle(data)
    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx, batch_size, n_sample)
        X, Y = batch[:,:-1], batch[:,-1]
        Pred = affine.forward(X)
        J = cost.forward(Y, Pred)

        dPred = cost.backward()
        affine.backward(dPred, lr)

        th_accum = np.hstack((th_accum, affine._Th))
        cost_list.append(J)

plt.scatter(x_data1,y_data,color = 'r')
tmp_Th = copy.deepcopy(affine._Th)
tmp_Th = tmp_Th.reshape((h_order+1))
tmp_y_data = []
for i in range(200):
    tmp_y_data.append(np.sum(x_data[i] * tmp_Th))
plt.plot(x_data1,tmp_y_data)
plt.show()

