import numpy as np

class CatCrossEnt:
    pass
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CatCrossEnt(Loss):
    def forward(self, y_pred, y_true):

        y_predClipped= np.clip(y_pred,1e-7, 1- 1e-7) # Avoid log(0)
        row_indices = range(len(y_pred))


        # 1 dimensional i.e [1,0,1]
        # Value at index i in y_true represents the target column in row i of y_pred
        if len(y_true.shape) ==1:
           res = y_predClipped[row_indices, y_true ]

        #Hot encoding i.e [[1,0,1][0,1,0][0,1,0]]
        elif len(y_true.shape) ==2:
            res = np.sum(y_predClipped * y_true,axis=1)  # colmn of targeted values (element-wise multiplication)

        return -np.log(res)




a = np.array([[1,2],[3,4],[5,6]])

print(a[range(3), [0,1,0]]) # firt arg must be nmbr of rows. Second must represent target for each row. Return array. of target values