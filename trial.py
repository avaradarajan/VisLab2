import numpy as np
import pandas as pd


test = np.array([[1, 1], [2, 2], [5, 6]])
print(test[0])
pdf = pd.DataFrame(data=test,
                   columns=['PCA1', 'PCA2'],
                   index=['A','B','C'])

print(pdf)
d = pdf.apply(lambda x: (x[0],x[1]), axis=1)
for i in d:
    print(i)

print(d)