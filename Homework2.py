import torch
import random
import timeit
import matplotlib.pyplot as plt
random.seed(737)
if torch.cuda.is_available():
  device = torch.device("cuda")
  print("cuda available");
else:
  device = torch.device("cpu")

#plot values
speedup=[]

for i in range(10,110,10):
    b = 0.
    X = [[random.random() for _ in range(110)]for _ in range(i)]
    W = [[random.random() for _ in range(i)]for _ in range(90)]
     # Initialize counter
     
    def plainMatrixMatrixMultiply(X, W, b):
        counter = 0 
        outputs = []
        for w in W:
            outputRow=[]
            for xCol in zip(*X):
                output = b
                for w_j, x_j in zip(w, xCol):
                    output += w_j * x_j
                outputRow.append(output)
            outputs.append(outputRow)
        return outputs
    res_plain_matrix = timeit.timeit ("plainMatrixMatrixMultiply(X, W, b)","from __main__ import plainMatrixMatrixMultiply; from __main__ import b,X,W",number=2000)
    #res_plain_matrix=1
    t_b = torch.tensor(b)
    X_t = torch.tensor(X)
    W_t = torch.tensor(W)
    res_pytorch_matrix = timeit.timeit ("W_t.matmul(X_t) + t_b","import torch; from __main__ import t_b,X_t,W_t",number=2000)
    #res_pytorch_matrix =1
    print(f'speedup = {res_plain_matrix / res_pytorch_matrix}')
    speedup.append(res_plain_matrix / res_pytorch_matrix)

plt.plot([10,20,30,40,50,60,70,80,90,100],speedup)
plt.ylabel('Speed improvement (plain/vec)')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.xlabel('value of m')
plt.show()