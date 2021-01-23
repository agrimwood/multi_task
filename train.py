"""Train the LSTM-net"""
import argparse
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import mse, MeanIoU
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd
import random
import os
import datetime
import mobnet
import data

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--learnrate', required=False, help='learning rate (required)', default=1e-4, type=float)
ap.add_argument('-b', '--batchsize', required=False, help='batch size (default 32)', type=int, default=3)
ap.add_argument('-e', '--epochs', required=False, help='epoch size (default 100)', default=200, type=int)
ap.add_argument('-d', '--data', required=False, help='directory containing images and labels folders (required)',default='/mnt/c/Experiments/ICR/png/proto',type=str)
ap.add_argument('-o', '--output', required=False, help='output directory for model, graph and epoch stats (required)',default='/mnt/c/Experiments/ICR/png',type=str)
ap.add_argument('-m', '--mirrored', required=False, help='use mirrored strategy for multi-gpu processing', default=False, type=bool)
args = vars(ap.parse_args())

# number of epochs?
#epochs = 3
epochs = args['epochs']

# log root directory?
#log_root = '/mnt/c/Users/rootm/wsl/logs'
log_root = args['output']  # path to output logs

# learning rate?
#learning_rate = 1e-3
learning_rate = args['learnrate']

# number of classes?
positions_class = 3
directions_class = 3

# data root directory?
#root_path = '/mnt/c/Experiments/ICR/png/proto/'
root_path = args['data']  # path to images, labels and rotImages

# batch_size?
#batch_size = 2
batch_size = args['batchsize']

# mirrored strategy?
dist_strat = args['mirrored']

######################################################################################
#epochs = 100
#log_root = '/mnt/c/Experiments/ICR/IJCARS/output'
#learning_rate = 1e-4
#root_path = '/mnt/c/Experiments/ICR/IJCARS/IJCARS_data'
#batch_size = 5
#dist_strat = False
######################################################################################

# autogenerate log directories
log_dir = os.path.join(log_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"))
os.makedirs(log_dir, exist_ok=True)

# target image size
img_width = 512
img_height = 512
imdimensions = (img_height,img_width)

# load ultrasound images, rotation images, and csv files containing labels
us_labels = os.path.join(root_path,'labels','labels_cons.csv')
df = pd.read_csv(us_labels, index_col='FileName')
img_path = os.path.join(root_path,'images')
msk_path = os.path.join(root_path,'masks')
# find unique patients in dataframe
fl_list = df.index.tolist()
pt_list = [fl_list[i][0:6] for i in range(len(fl_list))]
df['pt'] = pt_list
pts = list(set([fl_list[i][0:6] for i in range(len(fl_list))]))
######################################################################################

# specify loss functions and metrics so that the final loss is summed in quadrature
def dc_squared(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = 1 - (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return dice_coef

def cc_squared(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)**2


epsilon = backend_config.epsilon
def seg_bc_squared(y_true,y_pred):
    epsilon_ = constant_op.constant(epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
    bce = y_true * math_ops.log(y_pred + epsilon_)*0.9
    bce += (1 - y_true) * math_ops.log(1 - y_pred + epsilon_)*1.1
    return bce**2

pos_loss = "categorical_crossentropy"
dir_loss = "categorical_crossentropy"
seg_loss = SparseCategoricalCrossentropy(from_logits=True)

# establish distributed processing
if dist_strat is True:
    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        # metrics must be defined within strategy scope
        class MyMeanIOU(MeanIoU):
            def update_state(self, y_true, y_pred, sample_weight=None):
                return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
                
        miou = MyMeanIOU(num_classes=3)
        # create the model
        model = mobnet.LRASPP().build_model() 
        # compile the model
        model.compile(optimizer=RMSprop(lr=learning_rate), 
            loss={'prostate_out': pos_loss, 'direction_out': dir_loss, 'segment_out': seg_loss}, 
            metrics={'prostate_out': ['mse'], 'direction_out': ['mse'], 'segment_out': [miou]})

else:
    # define metric
    class MyMeanIOU(MeanIoU):
        def update_state(self, y_true, y_pred, sample_weight=None):
            return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
            
    miou = MyMeanIOU(num_classes=3)
    # create the model
    model = mobnet.LRASPP().build_model() 

    # compile the model
    model.compile(optimizer=RMSprop(lr=learning_rate), 
        loss={'prostate_out': pos_loss, 'direction_out': dir_loss, 'segment_out': seg_loss}, 
        metrics={'prostate_out': ['mse'], 'direction_out': ['mse'], 'segment_out': [miou]})

# write model graph to png file
model._layers = [
    layer for layer in model._layers if isinstance(layer, Layer)
]
plot_model(model, to_file=os.path.join(log_dir, 'mixed_Classifier.png'), show_shapes=True, show_layer_names=True)

# set checkpoints
checkpoint_filepath = os.path.join(log_dir, "model.best.hdf5")  # saves last model
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# set csv output
csv_filepath = os.path.join(log_dir, 'training.log')

#########################################################################################
# loop through epochs
k = 0
x = 0
for rng in range(epochs):
    # shuffle patient list and restart k-folding
    if k > len(pts)-5:
        k = 0
        random.shuffle(pts)

    # specify k-fold split for this epoch
    df_train = df[~df['pt'].isin(pts[k:k+5])]
    df_val = df[df['pt'].isin(pts[k:k+5])]
    train_generator = data.us_generator(dataframe=df_train, img_path=img_path, msk_path=msk_path, batch_size=batch_size)
    validation_generator = data.us_generator(dataframe=df_val, img_path=img_path, msk_path=msk_path, batch_size=batch_size)
    nb_train_samples = df_train.index.size
    nb_validation_samples = df_val.index.size

    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=x+1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        initial_epoch=x)

    # record prediction sample
    sample_predict.plot_mobnet_sample(model, df, root_path, imdimensions, log_dir, x)
    
    #write to log file
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0,'epoch',history.epoch[:])
    with open(csv_filepath, mode='a') as f:
        if rng == 0:
            hist_df.to_csv(f, header=True)
        else:
            hist_df.to_csv(f, header=False)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(log_dir, 'lstm_classifier.json'), 'w') as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights(os.path.join(log_dir, 'lstm_classifier.h5'))
    print('Saved model to disk: '+ log_dir)

    k += 6
    x += 1

# # specify k-fold split for this epoch
# train_generator = data.us_generator(df)
# nb_train_samples = df.index.size
# # set csv output
# csv_filepath2 = os.path.join(log_dir, 'trainingFinal.log')
# csv_logger2 = CSVLogger(csv_filepath2, separator=',', append=True)


# model.fit(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=x+1,
#     validation_data=None,
#     callbacks=[csv_logger2],
#     initial_epoch=x)

# # serialize model to JSON
# model_json = model.to_json()
# with open(os.path.join(log_dir, 'base_classifier.json'), 'w') as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(os.path.join(log_dir, 'base_classifier.h5'))
# print('Saved model to disk: '+ log_dir)






# %%
# import os
# os.chdir('/mnt/c/Users/rootm/wsl/scripts/tf2')
# import mobnet
# cls_out, seg1_out, seg2_out = mobnet.LRASPP().mobnet_backbone()
# pos_branch, pos_crossover = mobnet.LRASPP().lstm_head(input1=cls_out, label_category='pos_branch')
# dir_branch, _ = mobnet.LRASPP().lstm_head(input1=cls_out, input2=pos_crossover, label_category='dir_branch')
# seg_branch = mobnet.LRASPP().seg_head(input1=seg1_out, input2=seg2_out, num_classes=3)
# img_seq_input = Input(shape=(10, 512, 512, 3), batch_size=4, name='imagesS')
# opt_seq_input = Input(shape=(10, 6), batch_size=4, name='rotsS')
# model = Model(inputs=[img_seq_input,opt_seq_input],outputs=[pos_branch, dir_branch, seg_branch] )
# plot_model(model, to_file='/mnt/c/Users/rootm/x.png', show_shapes=True)
