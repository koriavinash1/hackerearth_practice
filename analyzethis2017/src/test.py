import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pickle
from sklearn import preprocessing

train = pd.read_csv('../dataset/Training_Dataset.csv')
sample = pd.read_csv('../dataset/Sample_Solution.csv')
train = train[train.mvar9 != 0]
train = train[train.mvar3 != 0]

def preProcessing(train, key):
	# 'mvar2', 'mvar3' check 
	if key == "train":
		x_train = train.drop(['cm_key', 'mvar16', 'mvar17', 'mvar18', 'mvar19', 'mvar20', 'mvar21', 'mvar22', 'mvar23', 'mvar24', 'mvar25', 'mvar26', 'mvar27', 'mvar28', 'mvar29', 'mvar30', 'mvar31', 'mvar32', 'mvar33', 'mvar34', 'mvar35','mvar1', 'mvar49', 'mvar50', 'mvar51', 'mvar12', 'mvar46', 'mvar47', 'mvar48'], axis=1)
	elif key == "eval":
		x_train = train.drop(['cm_key', 'mvar16', 'mvar17', 'mvar18', 'mvar19', 'mvar20', 'mvar21', 'mvar22', 'mvar23', 'mvar24', 'mvar25', 'mvar26', 'mvar27', 'mvar28', 'mvar29', 'mvar30', 'mvar31', 'mvar32', 'mvar33', 'mvar34', 'mvar35','mvar1', 'mvar12'], axis=1)
	x_train['mvar8'] = x_train['mvar8']-x_train['mvar7']
	x_train['mvar36'] = x_train['mvar36'] + x_train['mvar37'] +x_train['mvar38']+x_train['mvar39']
	#40-45 edit 
	x_train['mvar40'] =  x_train['mvar40']*x_train['mvar43']
	x_train['mvar41'] =  x_train['mvar41']*x_train['mvar44']
	x_train['mvar42'] =  x_train['mvar42']*x_train['mvar45']
	# family size
	x_train['mvar2'] = x_train['mvar2'] + 1
	x_train = x_train.drop(['mvar7','mvar37','mvar38', 'mvar39', 'mvar43', 'mvar44', 'mvar45'], axis=1)

	for c in x_train.columns:
		if c in ['mvar6', 'mvar8', 'mvar11', 'mvar36', 'mvar3', 'mvar9']:
			x_train[c] = (x_train[c] - min(x_train[c]))/(max(x_train[c]) - min(x_train[c]))

	normalizer = preprocessing.Normalizer().fit(x_train)
	return x_train

x_train = preProcessing(train, "train")
x_train.to_csv("down_input.csv", index=False) 
print(x_train.shape)
# 0&1 , 2&3, 4&5
_y = np.stack([train['mvar49'].values, train['mvar46'].values,train['mvar50'].values, train['mvar47'].values,train['mvar51'].values, train['mvar48'].values], 1)


y_train = []

for x in tqdm(_y):
	if x[2] == 1:
		y_train.append(1)
	elif x[4] == 1:
		y_train.append(2)
	elif x[0] == 1:
		y_train.append(0)
	else:
		if x[3] == 1:
			y_train.append(1)
		elif x[5] == 1:
			y_train.append(2)
		elif x[1] == 1:
			y_train.append(0)

# y_train = np.array(y_train)

# print(x_train.shape, y_train.shape)

split = 5000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

# print('Building DMatrix...')

# d_train = xgb.DMatrix(x_train, label=y_train)
# d_valid = xgb.DMatrix(x_valid, label=y_valid)
# del x_train, y_train

# print("Training ....")

# params = {}
# params['eta'] = 0.3
# params['objective'] = 'reg:logistic'
# params['eval_metric'] = 'mae'
# params['min_child_weight'] = 1
# params['max_depth'] = 24
# params['silent'] = 1

# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# clf = xgb.train(params, d_train, 1000000, watchlist, verbose_eval=10)

# del d_train, d_valid
# clf.save_model('0001.model')

leader_board = pd.read_csv("../dataset/Leaderboard_Dataset.csv")[:1000]

cm_key = leader_board['cm_key']
pdata = preProcessing(leader_board, "eval")

# clf0 = svm.SVR()
clf1 = RandomForestClassifier(max_depth = 14, random_state = 1)
clf2 = GaussianNB()
clf3 = LogisticRegression(random_state = 1)
eclf = VotingClassifier(iterations=500, estimators=[('lr', clf3), ('rf', clf1), ('gnb', clf2)], voting='soft')
eclf.fit(x_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(eclf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

print("validation score: "+str(loaded_model.score(x_valid, y_valid)))
print("On training score: "+str(loaded_model.score(x_train, y_train)))
asd = loaded_model.predict(pdata)
mapping = {'Credit':0, 'Elite':1, 'Supp':2}

# res = [np.argmax(i) for i in asd]
cardType = []
for i in asd:
	if i == 0: 
		cardType.append('Credit')
	elif i == 1:
		cardType.append('Elite')
	else :
		cardType.append('Supp')

sub = pd.DataFrame(data={'card':cardType, 'cm': cm_key})
sub = sub[['cm', 'card']]
sub.to_csv("Unity_IITMadras_5.csv", index=False) 