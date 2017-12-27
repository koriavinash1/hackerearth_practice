from keras.models import load_model
import numpy as np
import pandas as pd

test = pd.read_csv("../dataset/test.csv")
# test = pd.read_csv("../dataset/test.csv")
test.loc[:,'AngleOfSign'] *= 1.0/360.0

#renaming column
# test.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)
test_Ids = test['Id']

test.drop(['Id', 'SignWidth', 'SignHeight'], inplace=True, axis=1)
test_data = test.as_matrix()

ntest = []
for cp, an, asp in zip(test_data.T[0], test_data.T[1], test_data.T[2]):
    if cp == 'Front':
        ntest.append([0, an, asp])
    elif cp == 'Right':
        ntest.append([1, an, asp])
    elif cp == 'Left':
        ntest.append([2, an, asp])
    else:
        ntest.append([3, an, asp])

ntest = np.array(ntest)

batch_size= 128

model = load_model("./results/best_model/final_model.hdf5")
predictions = np.array(model.predict(ntest))

columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=predictions, columns=columns)
sub['Id'] = test_Ids
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("MLP_results.csv", index=False) #99.8XXX