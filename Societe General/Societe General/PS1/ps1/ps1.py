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
    # convert all cost to USD
    # array(['CHF', 'EUR', 'GBP', 'JPY', 'USD'], dtype=object)
    
    # CHF to USD
    df['sold'][df['currency']=='CHF'] = df['sold']*1.01
    # EUR to USD
    df['sold'][df['currency']=='EUR'] = df['sold']*1.19
    # GBP to USD
    df['sold'][df['currency']=='GBP'] = df['sold']*1.34
    # JPY to USD
    df['sold'][df['currency']=='JPY'] = df['sold']*0.01
    
    # remove desk_id, office_id, pf_id
    # think about indicator_code, hedge_value, status
    df = df.drop(["portfolio_id","desk_id", "office_id","indicator_code","hedge_value","status", "currency"], axis=1) 
    
    # vectorise type and pf_category
    vec = 1
    for i in np.unique(df['type']):
        df['type'][df['type'] == i] = vec
        vec = vec + 1
        
    vec = 1
    for i in np.unique(df['pf_category']):
        df['pf_category'][df['pf_category'] == i] = vec
        vec = vec + 1
        
    vec = 26
    for i in np.unique(df['country_code']):
        df['country_code'][df['country_code'] == i] = vec
        vec = vec - 1
    
    # how to change datatypes in df
    df['type'] = pd.to_numeric(df['type'],errors='coerce')
    df['country_code'] = pd.to_numeric(df['country_code'],errors='coerce')
    df['pf_category'] = pd.to_numeric(df['pf_category'],errors='coerce')
    
    # mean sd normalization for sold col...
    mean = np.mean(df['sold'])
    sd = np.sqrt(np.var(df['sold']))
    df['sold'] = (df['sold']-mean)/sd
    
    # normalization for bought col...
    mean = np.mean(df['bought'])
    sd = np.sqrt(np.var(df['bought']))
    df['bought'] = (df['bought']-mean)/sd
    
    # normalization for libor_rate
    mean = np.mean(df['libor_rate'])
    sd = np.sqrt(np.var(df['libor_rate']))
    df['libor_rate'] = (df['libor_rate']-mean)/sd
    return df

print(dataset_train_validate)

processed_data = preprocessing(dataset_train_validate)
print(processed_data)

train_set = processed_data[:6300]
test_set = processed_data[6300:]

# for visualization
plt.hist(processed_data['euribor_rate'])
plt.show()

plt.plot(processed_data['bought'])
plt.show()

y_train = train_set['return'].values
x_train = train_set.drop(['return'], axis=1)

y_valid = test_set['return'].values
x_valid = test_set.drop(['return'], axis=1)

print x_train.shape, y_train.shape,x_valid.shape, y_valid.shape

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

"""
Training ...
[0]	train-mae:0.47181	valid-mae:0.485401
Multiple eval metrics have been passed: 'valid-mae' will be used for early stopping.

Will train until valid-mae hasn't improved in 100 rounds.
[10]	train-mae:0.385577	valid-mae:0.396995
[20]	train-mae:0.315119	valid-mae:0.324718
[30]	train-mae:0.257551	valid-mae:0.265484
[40]	train-mae:0.210509	valid-mae:0.217048
[50]	train-mae:0.172071	valid-mae:0.177478
[60]	train-mae:0.140675	valid-mae:0.145156
[70]	train-mae:0.115022	valid-mae:0.118641
[80]	train-mae:0.094054	valid-mae:0.096965
[90]	train-mae:0.076928	valid-mae:0.079248
[100]	train-mae:0.062938	valid-mae:0.064746
[110]	train-mae:0.051515	valid-mae:0.052851
[120]	train-mae:0.042185	valid-mae:0.04309
[130]	train-mae:0.034562	valid-mae:0.035155
[140]	train-mae:0.028333	valid-mae:0.028608
[150]	train-mae:0.023262	valid-mae:0.023259
[160]	train-mae:0.019136	valid-mae:0.018953
[170]	train-mae:0.015781	valid-mae:0.01542
[180]	train-mae:0.013041	valid-mae:0.012537
[190]	train-mae:0.010807	valid-mae:0.010346
[200]	train-mae:0.00899	valid-mae:0.008702
[210]	train-mae:0.007517	valid-mae:0.007428
[220]	train-mae:0.006321	valid-mae:0.006448
[230]	train-mae:0.005364	valid-mae:0.00571
[240]	train-mae:0.0046	valid-mae:0.005124
[250]	train-mae:0.003984	valid-mae:0.004673
[260]	train-mae:0.003487	valid-mae:0.004401
[270]	train-mae:0.003089	valid-mae:0.004225
[280]	train-mae:0.002771	valid-mae:0.004086
[290]	train-mae:0.002527	valid-mae:0.003976
[300]	train-mae:0.002333	valid-mae:0.003895
[310]	train-mae:0.002175	valid-mae:0.003832
[320]	train-mae:0.002053	valid-mae:0.003785
[330]	train-mae:0.001963	valid-mae:0.003758
[340]	train-mae:0.001888	valid-mae:0.00374
[350]	train-mae:0.001829	valid-mae:0.003727
[360]	train-mae:0.00178	valid-mae:0.003717
[370]	train-mae:0.001745	valid-mae:0.003708
[380]	train-mae:0.001716	valid-mae:0.003701
[390]	train-mae:0.001691	valid-mae:0.003697
[400]	train-mae:0.001669	valid-mae:0.003691
[410]	train-mae:0.001647	valid-mae:0.003684
[420]	train-mae:0.001626	valid-mae:0.003679
[430]	train-mae:0.001614	valid-mae:0.00368
[440]	train-mae:0.001598	valid-mae:0.003677
[450]	train-mae:0.001585	valid-mae:0.003673
[460]	train-mae:0.001571	valid-mae:0.003671
[470]	train-mae:0.001557	valid-mae:0.00367
[480]	train-mae:0.001545	valid-mae:0.003671
[490]	train-mae:0.001531	valid-mae:0.003668
[500]	train-mae:0.001521	valid-mae:0.003667
[510]	train-mae:0.001511	valid-mae:0.003665
[520]	train-mae:0.001506	valid-mae:0.003667
[530]	train-mae:0.001499	valid-mae:0.003668
[540]	train-mae:0.001491	valid-mae:0.003666
[550]	train-mae:0.001485	valid-mae:0.003668
[560]	train-mae:0.001477	valid-mae:0.003669
[570]	train-mae:0.001472	valid-mae:0.003669
[580]	train-mae:0.001465	valid-mae:0.003672
[590]	train-mae:0.00146	valid-mae:0.003673
[600]	train-mae:0.001455	valid-mae:0.003673
Stopping. Best iteration:
[509]	train-mae:0.001512	valid-mae:0.003665
"""
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
