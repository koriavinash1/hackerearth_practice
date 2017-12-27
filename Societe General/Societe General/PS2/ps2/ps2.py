import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import gc

dataset_train_validate = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

# train: 6300 rest for validation
# split after preprocessing
# train = dataset_train_validate[]

# preprocessing 
def preprocessing(df):
    # remove str cat variables..
    for i in range(18):
        st = 'cat_var_' + str(i+1)
        df = df.drop([st], axis = 1)
    
    # mean sd normalization for sold col...
    for i in range(7):
        st = 'num_var_' + str(i+1)
        mean = np.mean(df[st])
        sd = np.sqrt(np.var(df[st]))
        df[st] = (df[st] - mean) / sd
    
    df = df.drop(['transaction_id'], axis=1)        
    return df

print(dataset_train_validate)

processed_data = preprocessing(dataset_train_validate)
print(processed_data)

train_set = processed_data[:300000]
test_set = processed_data[300000:]

# visualization
plt.plot(processed_data['num_var_6'])
plt.show()

y_train = train_set['target'].values
x_train = train_set.drop(['target'], axis=1)

y_valid = test_set['target'].values
x_valid = test_set.drop(['target'], axis=1)

print x_train.shape, y_train.shape,x_valid.shape, y_valid.shape

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

# free memory
del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 16
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 100000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid


print('Building test set ...')

submission = {}

submission['portfolio_id'] = dataset_test['portfolio_id'].values
x_test = preprocessing(dataset_test)

d_test = xgb.DMatrix(x_test)
p_test = clf.predict(d_test)

submission['return'] = p_test

print submission

# df construction
sub_df = pd.DataFrame(data=submission, index=np.arange(len(submission['return'])))

# create submission.csv file
print('Writing csv ...')
sub_df.to_csv('submission.csv', index=False, float_format='%.4f')

# done........................................................................
