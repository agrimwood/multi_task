"""Build a MobileNet backbone using keras applications"""
#%%
import tensorflow as tf
from tensorflow.keras import applications as mn
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input, Flatten, Lambda, Reshape, TimeDistributed, LSTM, Dropout, Dense, concatenate, Multiply, ConvLSTM2D, Conv2D, AveragePooling2D, Add, Conv2DTranspose, BatchNormalization, Activation, ReLU
from tensorflow.image import resize
from tensorflow.keras.utils import plot_model

# %%
class unet():
    def __init__(self, net_type='MobileNetV2', trainable=False, batch_size=None, frames=10, lstm_size=256, num_classes=3, seg_classes=1,im_dimensions=[640,480,3]):
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
        opt_input = Input(shape=(6), batch_size=batch_size, name='rots')
        img_seq_input = Input(shape=seq_shape, batch_size=batch_size, name='imagesS')
        opt_seq_input = Input(shape=(frames, 6), batch_size=batch_size, name='rotsS')
        self.backbone_input=backbone_input
        self.opt_input=opt_input
        self.img_seq_input=img_seq_input
        self.opt_seq_input=opt_seq_input

    def _backbone(self):
        ### set function params
        net_type=self.net_type
        trainable=self.trainable
        backbone_input=self.backbone_input
        #img_seq_input=self.img_seq_input
        #opt_seq_input=self.opt_seq_input
        #batch_size=self.batch_size
        #frames=self.frames
        #lstm_size=self.lstm_size
        #num_classes=self.num_classes


        ### set inputs with a fixed batch size
        # backbone_input = Input(shape=(512,512,3), name='images', batch_size=4)
        # img_seq_input = Input(shape=(10, 512, 512, 3), name='imagesS', batch_size=4)
        # opt_seq_input = Input(shape=(10, 6), name='rotsS', batch_size=4)

        ### select model type
        if net_type is 'MobileNetV2':
            x = mn.mobilenet_v2.MobileNetV2(
                input_tensor=backbone_input,
                include_top=False,
                pooling='avg',
                weights='imagenet'
                )
            layer_names = [
                'block_1_expand_relu', 
                'block_3_expand_relu', 
                'block_6_expand_relu', 
                'block_13_expand_relu', 
                'block_16_project',
                'global_average_pooling2d']
            
        # specify whether weights are trainable
        layers = [x.get_layer(name).output for name in layer_names]

        # create backbone
        x = Model(inputs=backbone_input,outputs=layers)

        # specify whether weights are trainable
        x.trainable = trainable
        return x

    ### upsampling module for decoder
    def _upsample(self,filters,size):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            Conv2DTranspose(filters, size, strides=2, 
            padding='same',
            kernel_initializer=initializer,
            use_bias=False)
        )
        result.add(BatchNormalization())
        result.add(ReLU())
        return result
    
    ### unet decoder
    def _unet(self,encoder_outputs,skip_stack):
        # set segmentation classes
        seg_classes=self.seg_classes
        # add decoder layers
        x=encoder_outputs[-1]
        skips = reversed(encoder_outputs[:-1])
        for up, skip in zip(skip_stack,skips):
            x = up(x)
            x = concatenate(inputs=[x,skip], axis=-1)

        x = Conv2DTranspose(seg_classes,3,strides=2,padding='same',activation='softmax')(x) # 1/2 -> 1
        return x 
        
    def _cls_head(self, cls_out, finalAct):
        ### create classification heads
        # PROSTATE BRANCH parameters
        input1=concatenate(inputs=[cls_out,self.opt_input],axis=-1)
        num_classes=self.num_classes
        label_category='prostate'
        fc0_size = 256
        fc1_size = fc0_size / 2
        fc2_size = fc1_size / 2
        final_fc_name = label_category + "_fc2"
        output_name = label_category + "_out"

        # specify network structure
        x = Dense(fc0_size)(input1)
        x = Dropout(0.5)(x)
        x = Dense(fc1_size)(x)
        x = Dropout(0.5)(x)
        x_out = Dense(fc2_size, name=final_fc_name)(x)
        x = Dropout(0.5)(x_out)
        prostate_out = Dense(num_classes, activation=finalAct, name=output_name)(x)

        # DIRECTION BRANCH parameters
        label_category='direction'
        final_fc_name = label_category + "_fc2"
        output_name = label_category + "_out"

        # specify network structure
        x = Dense(fc0_size)(input1)
        x = Dropout(0.5)(x)
        x = Dense(fc1_size)(x)
        x = Dropout(0.5)(x)
        x = Dense(fc2_size)(x)
        x = Concatenate()([x,x_out])
        x = Dense(fc2_size, name=final_fc_name)(x)
        x = Dropout(0.5)(x)
        direction_out = Dense(num_classes, activation=finalAct, name=output_name)(x)
        return prostate_out, direction_out

    def build_model(self, seg_classes=1, cls_act='softmax'):
        #specify upsampling blocks
        skip_stack = [
            self._upsample(512,3),  #1/32  
            self._upsample(256,3),  #1/16
            self._upsample(128,3),  #1/8
            self._upsample(64,3),    #1/4
            ]
        # build backbone encoder
        x = self._backbone()
        # build decoder head
        seg_inputs = x.outputs[:-1]
        segment_out = self._unet(seg_inputs,skip_stack)
        # build classifier heads
        cls_inputs = x.outputs[-1]
        prostate_out, direction_out = self._cls_head(cls_inputs, cls_act)
        # build model
        model = Model(inputs=[self.backbone_input,self.opt_input], outputs=[segment_out,prostate_out,direction_out])
        return model

###############################################################################
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
