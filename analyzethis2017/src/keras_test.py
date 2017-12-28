import pandas as pd
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import numpy as np

train = pd.read_csv('../dataset/Training_Dataset.csv')
sample = pd.read_csv('../dataset/Sample_Solution.csv')

train = train[train.mvar9 != 0]
train = train[train.mvar3 != 0]

def preProcessing(train, key):
	if key == "train":
		x_train = train.drop(['cm_key', 'mvar16', 'mvar17', 'mvar18', 'mvar19', 'mvar20', 'mvar21', 'mvar22', 'mvar23', 'mvar24', 'mvar25', 'mvar26', 'mvar27', 'mvar28', 'mvar29', 'mvar30', 'mvar31', 'mvar32', 'mvar33', 'mvar34', 'mvar35','mvar1', 'mvar49', 'mvar50', 'mvar51', 'mvar12', 'mvar46', 'mvar47', 'mvar48'], axis=1)
	elif key == "eval":
		x_train = train.drop(['cm_key', 'mvar16', 'mvar17', 'mvar18', 'mvar19', 'mvar20', 'mvar21', 'mvar22', 'mvar23', 'mvar24', 'mvar25', 'mvar26', 'mvar27', 'mvar28', 'mvar29', 'mvar30', 'mvar31', 'mvar32', 'mvar33', 'mvar34', 'mvar35','mvar1', 'mvar12'], axis=1)
	x_train['mvar8'] = x_train['mvar8']-x_train['mvar7']
	x_train['mvar36'] = x_train['mvar36'] + x_train['mvar37'] +x_train['mvar38']+x_train['mvar39']
	x_train = x_train.drop(['mvar7','mvar37','mvar38', 'mvar39'], axis=1)
	for c in x_train.columns:
		x_train[c] = x_train[c]/max(x_train[c])

	return x_train

x_train = preProcessing(train, "train")
y_train = np.stack([train['mvar49'].values * train['mvar46'].values,train['mvar50'].values*train['mvar47'].values,train['mvar51'].values*train['mvar48'].values], 1)

train_data = x_train.as_matrix()
print train_data.shape

batch_size = 128

_input = Input(shape=(19, ), name='model_input')
x = Dense(2048, activation='tanh', name='fc1')(_input)
x = Dropout(0.2)(x)
x = Dense(1024, activation='tanh', name='fc2')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='tanh', name='fc3')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu', name='fc4')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu', name='fc5')(x)
x = Dropout(0.2)(x)
output = Dense(3, activation='softmax', name='predictions')(x)
model = Model(input=_input, output=output)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

# checkpointer = ModelCheckpoint(filepath="./results/best_model/fn_model.{epoch:02d}-{val_acc:.2f}.hdf5", verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='max', period=1)
# tf_board = TensorBoard(log_dir='./results/logs', histogram_freq=0, write_graph=True, write_images=True)
# csv_logger = CSVLogger('./results/training.log')
# early_stopping = EarlyStopping(monitor='val_loss', patience=12)
# , callbacks=[early_stopping, checkpointer, tf_board, csv_logger]
model.fit(train_data, y_train, batch_size=batch_size, nb_epoch=80, validation_split=0.08)
model.save("final_model.hdf5")

