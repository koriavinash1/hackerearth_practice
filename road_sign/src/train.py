import pandas as pd
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import numpy as np

train = pd.read_csv("../dataset/train.csv").iloc[500:]
# test = pd.read_csv("../dataset/test.csv")

# preprocessing....
train.loc[:,'AngleOfSign'] *= 1.0/360.0
# test.loc[:,'AngleOfSign'] *= 1.0/360.0

#renaming column
train.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)
train_labels = train['Target']

train.drop(['Id', 'SignWidth', 'SignHeight', 'Target'], inplace=True, axis=1)
train_data = train.as_matrix()

ntrain, nlabels = [], []
for cp, an, asp, label in zip(train_data.T[0], train_data.T[1], train_data.T[2], train_labels.as_matrix().T):
    if cp == 'Front':
        ntrain.append([0, an, asp])
        nlabels.append([1.0, 0.0, 0.0, 0.0])
    elif cp == 'Right':
        ntrain.append([1, an, asp])
        nlabels.append([0.0, 0.0, 0.0, 1.0])
    elif cp == 'Left':
        ntrain.append([2, an, asp])
        nlabels.append([0.0, 1.0, 0.0, 0.0])
    else:
        ntrain.append([3, an, asp])
        nlabels.append([0.0, 0.0, 1.0, 0.0])

ntrain = np.array(ntrain)
nlabels = np.array(nlabels)

batch_size = 128

_input = Input(shape=(3,), name='model_input')
x = Dense(512, activation='sigmoid', name='fc1')(_input)
output = Dense(4, activation='softmax', name='predictions')(x)
model = Model(input=_input, output=output)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="./results/best_model/fn_model.{epoch:02d}-{val_acc:.2f}.hdf5", verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='max', period=1)
tf_board = TensorBoard(log_dir='./results/logs', histogram_freq=0, write_graph=True, write_images=True)
csv_logger = CSVLogger('./results/training.log')
early_stopping = EarlyStopping(monitor='val_loss', patience=12)

model.fit(ntrain, nlabels, batch_size=batch_size, nb_epoch=80, validation_split=0.08, callbacks=[early_stopping, checkpointer, tf_board, csv_logger])
model.save("./results/best_model/final_model.hdf5")