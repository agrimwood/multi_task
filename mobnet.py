"""Build a MobileNet backbone using keras applications"""
#%%
import tensorflow as tf
from tensorflow.keras import applications as mn
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input, Flatten, Lambda, Reshape, TimeDistributed, LSTM, Dropout, Dense, concatenate, Multiply, ConvLSTM2D, Conv2D, AveragePooling2D, Add, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.image import resize
from tensorflow.keras.utils import plot_model

# %%
class LRASPP():
    def __init__(self, net_type='MobileNetV2', trainable=False, batch_size=None, frames=10, lstm_size=256, num_classes=3, seg_classes=2, im_dimensions=[512,512,3]):
        self.net_type = net_type
        self.trainable = trainable
        self.batch_size = batch_size
        self.frames = frames
        self.lstm_size = lstm_size
        self.num_classes=num_classes
        self.seg_classes=seg_classes
        
        ### set inputs
        seq_shape=[frames]
        seq_shape.extend(im_dimensions)
        backbone_input = Input(shape=im_dimensions, batch_size=batch_size, name='images')
        img_seq_input = Input(shape=seq_shape, batch_size=batch_size, name='imagesS')
        opt_seq_input = Input(shape=(frames, 6), batch_size=batch_size, name='rotsS')
        self.backbone_input=backbone_input
        self.img_seq_input=img_seq_input
        self.opt_seq_input=opt_seq_input

    def _backbone(self):
        ### set function params
        net_type=self.net_type
        trainable=self.trainable
        #batch_size=self.batch_size
        #frames=self.frames
        #lstm_size=self.lstm_size
        #num_classes=self.num_classes
        backbone_input=self.backbone_input
        #img_seq_input=self.img_seq_input
        #opt_seq_input=self.opt_seq_input


        ### set inputs with a fixed batch size
        # backbone_input = Input(shape=(512,512,3), name='images', batch_size=4)
        # img_seq_input = Input(shape=(10, 512, 512, 3), name='imagesS', batch_size=4)
        # opt_seq_input = Input(shape=(10, 6), name='rotsS', batch_size=4)

        ### select model type
        if net_type is 'MobileNetV2':
            seg_layer1 = 'block_6_expand_relu'
            seg_layer2 = 'block_13_expand_relu'
            cls_layer = 'global_average_pooling2d'
            x = mn.mobilenet_v2.MobileNetV2(include_top=False,pooling='avg',weights='imagenet',input_tensor=backbone_input)
        
        ### is this necessary???
        #x.inputs = x.get_layer('images').input

        ### build multi-output sequence backbone
        for layer in x.layers:
            layer.trainable = trainable

        # specify backbone outputs
        output_cls = x.get_layer(cls_layer).output
        output_seg1 = x.get_layer(seg_layer1).output
        output_seg2 = x.get_layer(seg_layer2).output
        x.orig_shapes={'cls': output_cls.get_shape().as_list(), 'seg1': output_seg1.get_shape().as_list(), 'seg2': output_seg2.get_shape().as_list()}

        # flatten and concatenate the 3 outputs to feed into TimeDistributed
        a = Flatten()(output_cls)
        b = Flatten()(output_seg1)
        c = Flatten()(output_seg2)
        x.flat_shapes={'cls': a.get_shape().as_list(), 'seg1': b.get_shape().as_list(), 'seg2': c.get_shape().as_list()}
        x.outputs = concatenate(inputs=[a, b, c], axis=-1)
        return x

    def _reshape_outputs(self, x, flat_shapes, orig_shapes):        
        frames=self.frames

        # split the flattened output, concatenate optical tracking input and reshape features
        idx1 = flat_shapes['cls'][1]
        idx2 = idx1 + flat_shapes['seg1'][1]
        idx3 = idx2 + flat_shapes['seg2'][1]
        cls_out = concatenate(inputs=[x[..., 0:idx1], self.opt_seq_input], axis=-1)
        recon_shape1 = [frames]
        recon_shape1.extend(orig_shapes['seg1'][1:])
        seg1_out = Reshape(recon_shape1)(x[..., idx1:idx2])
        recon_shape2 = [frames]
        recon_shape2.extend(orig_shapes['seg2'][1:])
        seg2_out = Reshape(recon_shape2)(x[..., idx2:idx3])
        return cls_out, seg1_out, seg2_out

    def _lstm_head(self, cls_out, finalAct):
        ### create LSTM heads
        ## LSTM 1
        # parameters
        input1=cls_out
        num_classes=self.num_classes
        lstm_size=self.lstm_size
        label_category='prostate'

        # establish layer parameters
        if lstm_size < 16:
            lstm_size = 16

        fc1_size = lstm_size / 2
        fc2_size = fc1_size / 2
        final_fc_name = label_category + "_fc2"
        output_name = label_category + "_out"

        # specify network structure
        x = LSTM(lstm_size)(input1)
        x = Dropout(0.5)(x)
        x = Dense(fc1_size)(x)
        x = Dropout(0.5)(x)
        x_out = Dense(fc2_size, name=final_fc_name)(x)
        x = Dropout(0.5)(x_out)
        prostate_out = Dense(num_classes, activation=finalAct, name=output_name)(x)

        ## LSTM 2
        # parameters
        input2=x_out
        label_category='direction'

        # establish layer parameters
        lstm_size = lstm_size
        if lstm_size < 16:
            lstm_size = 16

        fc1_size = lstm_size / 2
        fc2_size = fc1_size / 2
        final_fc_name = label_category + "_fc2"
        output_name = label_category + "_out"

        # specify network structure
        x = LSTM(lstm_size)(input1)
        x = Dropout(0.5)(x)
        x = Dense(fc1_size)(x)
        x = Dropout(0.5)(x)
        x = Dense(fc2_size)(x)
        x = Concatenate()([x,input2])
        x = Dense(fc2_size, name=final_fc_name)(x)
        x = Dropout(0.5)(x)
        direction_out = Dense(num_classes, activation=finalAct, name=output_name)(x)
        return prostate_out, direction_out

    def _seg_head(self, seg1_out, seg2_out):
        ### create segmentation head
        # specify parameters
        input1 = seg1_out
        input2 = seg2_out
        seg_classes=self.seg_classes+1

        # Pre-processing LSTM layers - if memory is an issue, replace with Conv2D of last slice
        input1 = ConvLSTM2D(filters=input1.get_shape()[-1], kernel_size=(1,1))(input1)
        #input1 = Lambda(lambda x: resize(x,[128,128]), name='resize_input1')(input1)
        input2 = ConvLSTM2D(filters=input2.get_shape()[-1], kernel_size=(1,1))(input2)
        #input2 = Lambda(lambda x: resize(x,[64,64]), name='resize_input2')(input2)

        # branch 1
        x1 = Conv2D(filters=128, kernel_size=(1, 1))(input2)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        h1, w1 = x1.get_shape()[1:3]

        # branch 2
        x2 = AveragePooling2D(pool_size=(25, 25), strides=(7, 7))(input2)
        x2 = Conv2D(128, (1, 1))(x2)
        x2 = Activation('sigmoid')(x2)
        x2 = resize(x2,[h1, w1])

        # branch 3
        x3 = Conv2D(filters=seg_classes, kernel_size=(1, 1))(input1)
        h3, w3 = x3.get_shape()[1:3]

        # merge and continue branch 1
        x1 = Multiply()([x1,x2])
        x1 = resize(x1,[h3, w3])
        x1 = Conv2D(filters=seg_classes, kernel_size=(1, 1))(x1)
        x1 = Add()([x1, x3])
        #x1 = Lambda(lambda x: resize(x,[512,512]))(x1)
        segment_out = Conv2DTranspose(
            filters=seg_classes, 
            kernel_size=7, 
            strides=8, 
            padding='same', 
            name='segment_out'
            )(x1)
        return segment_out

    def build_model(self, cls_act='softmax'):
        ### cls_act = 'softmax' ### normal for one hot encoding
        
        # build backbone model and wrap for image sequences
        x = self._backbone()
        flat_shapes=x.flat_shapes
        orig_shapes=x.orig_shapes
        x = Model(x.inputs,x.outputs) 
        model = TimeDistributed(x)(self.img_seq_input)
        # reshape outputs before passing to heads
        cls_out, seg1_out, seg2_out = self._reshape_outputs(model,flat_shapes,orig_shapes)
        # pass class output to LSTM heads
        prostate_out, position_out = self._lstm_head(cls_out, cls_act)
        # pass seg outputs to segmentation head
        segmentation_out = self._seg_head(seg1_out, seg2_out)

        model = Model(inputs=[self.img_seq_input, self.opt_seq_input], outputs=[prostate_out,position_out,segmentation_out])

        return model

###############################################################################
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

        
# %%
