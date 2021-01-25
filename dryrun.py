import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import mse, MeanIoU
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
import numpy as np
import pandas as pd
import random
import os
import datetime
import mobnet
import data
import sample_predict

epochs = 1500
log_root = '/raid/candi/alex/IJCARS_output'
learning_rate = 1e-4
root_path = '/raid/candi/alex/IJCARS_data'
batch_size = 15
dist_strat = False
# autogenerate log directories
log_dir = os.path.join(log_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"))
os.makedirs(log_dir, exist_ok=True)
# number of classes?
positions_class = 3
directions_class = 3
# target image size?
img_width = 512
img_height = 512
imdimensions = (img_height,img_width)
######################################################################################
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

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1", "/gpu:2"])
#with mirrored_strategy.scope():
pos_loss = "categorical_crossentropy"
dir_loss = "categorical_crossentropy"
seg_loss = SparseCategoricalCrossentropy(from_logits=True)
class MyMeanIOU(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
        
miou = MyMeanIOU(num_classes=3)
# create the model
model = mobnet.LRASPP(im_dimensions=[img_height,img_width,3]).build_model() 


# compile the model
model.compile(optimizer=RMSprop(lr=learning_rate), 
    loss={'prostate_out': pos_loss, 'direction_out': dir_loss, 'segment_out': seg_loss}, 
    metrics={'prostate_out': ['mse'], 'direction_out': ['mse'], 'segment_out': [miou]})

plot_model(model, to_file=os.path.join(log_dir, 'mixed_Classifier.png'), show_shapes=True, show_layer_names=True)

# set checkpoints
checkpoint_filepath = os.path.join(log_dir, "model.best.hdf5")  # saves last model
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# set scheduler callback
def scheduler(epoch, lr):
    if epoch < 45:
        return lr
    elif epoch < 55:
        return lr * tf.math.exp(-0.2)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# set csv output
csv_filepath = os.path.join(log_dir, 'training.log')

# loop through epochs
val_losses = 1
k = 0
nx = 0
for rng in range(epochs):
    # shuffle patient list and restart k-folding
    if k > len(pts)-7:
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
        x=train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nx+1,
	validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        initial_epoch=nx,
        callbacks=[callback])

    # record prediction sample
    sample_predict.plot_mobnet_sample(model, df, root_path, imdimensions, log_dir, nx)
    
    ##write to log file
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0,'epoch',history.epoch[:])
    with open(csv_filepath, mode='a') as f:
        if rng == 0:
            hist_df.to_csv(f, header=True, line_terminator='\n')
        else:
            hist_df.to_csv(f, header=False, line_terminator='\n')

    
    if hist_df.val_loss < val_losses:
        val_losses = hist_df.val_loss
        ## model to JSON
        model_json = model.to_json()
        with open(os.path.join(log_dir, 'lstm_classifier.json'), 'w') as json_file:
            json_file.write(model_json)
    
        # weights to HDF5
        model.save_weights(os.path.join(log_dir, 'lstm_classifier.h5'))
        print('Saved model to disk: '+ log_dir)


    k += 6
    nx += 1
