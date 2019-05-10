import numpy as np
from alexnet import alexnet
WIDTH = 3
HEIGHT = 1
LR = 1e-3
EPOCHS = 5

MODEL_NAME = 'PUBG_PLAYZONE_PREDICTION.model'


model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('new_training_data.npy')
train = train_data[:int(len(train_data)*0.9)]
test = train_data[int(len(train_data)*0.9):]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

##print(Y)
##input()
test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)


#tensorboard --logdir=foo:F:/PUBG/log
