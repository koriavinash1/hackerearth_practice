import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn import preprocessing
import numpy as np 

tdata = pd.read_csv("../dataset/train.csv")
train = tdata.iloc[500:]
validate = tdata.iloc[:500]
test = pd.read_csv("../dataset/test.csv")
test.loc[:,'AngleOfSign'] *= 1.0/360.0
test_id = test['Id']

train['DetectedCamera'].value_counts()
train.loc[:,'AngleOfSign'] *= 1.0/360.0
validate.loc[:,'AngleOfSign'] *= 1.0/360.0
test.loc[:,'AngleOfSign'] *= 1.0/360.0

#encode as integer
mapping = {'Front':0, 'Right':1, 'Left':2, 'Rear':3}
train = train.replace({'DetectedCamera': mapping})
validate = validate.replace({'DetectedCamera': mapping})
test = test.replace({'DetectedCamera': mapping})

#renaming column
train.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)
validate.rename(columns = {'SignFacing (Target)': 'Target'}, inplace = True)
test.rename(columns = {'SignFacing (Target)': 'Target'}, inplace = True)

#encode Target Variable based on sample submission file
mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'Target': mapping})
validate = validate.replace({'Target': mapping})

#target variable
y_train = train['Target']
y_validate = validate['Target']
test_id = test['Id']

y_validate.as_matrix()
ny_validate = []
for y in y_validate:
    if y == 0:
        ny_validate.append([1.0, 0.0, 0.0, 0.0])
    elif y == 1:
        ny_validate.append([0.0, 1.0, 0.0, 0.0])
    elif y == 2:
        ny_validate.append([0.0, 0.0, 1.0, 0.0])
    else:
        ny_validate.append([0.0, 0.0, 0.0, 1.0])
ny_validate = np.array(ny_validate)

train.drop(['Target','Id'], inplace=True, axis=1)
validate.drop(['Target','Id'], inplace=True, axis=1)
test.drop('Id',inplace=True,axis=1)

clf = RandomForestClassifier(n_estimators=5000, max_features=5, min_samples_split=5, oob_score=True)
clf.fit(train, y_train)
pred = clf.predict_proba(test)

columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("random_forest_results.csv", index=False)