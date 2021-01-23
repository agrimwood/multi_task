import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing import image as krs
from tensorflow.keras import applications as mn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
        
def unet_sample(dataframe, root_dir, imdimensions):
    # discard first 9 slices from each sequence
    dataframe['slice'] = dataframe.index.str[-7:-4].astype(int)
    dataframe[dataframe['slice'] >=10]
    #collect a random sample from each class
    p1 = dataframe.loc[dataframe.position.eq('outside')].sample()
    p2 = dataframe.loc[dataframe.position.eq('periphery')].sample()
    p3 = dataframe.loc[dataframe.position.eq('centre')].sample()
    d1 = dataframe.loc[dataframe.direction.eq('left')].sample()
    d2 = dataframe.loc[dataframe.direction.eq('stop')].sample()
    d3 = dataframe.loc[dataframe.direction.eq('right')].sample()
    lst = [p1,p2,p3,d1,d2,d3]
    
    #load sample inputs and masks
    y1=[]
    y1_cons=[]
    y2=[]
    y2_cons=[]
    rotfeatures=[]
    i=0
    for sample in lst:
        # record consensus labels
        y1.append(sample.position[0])
        y2.append(sample.direction[0])

        # record consenus
        consensus1=sample[['outside','periphery','centre']].values
        y1_cons.append(consensus1.max())
        consensus2=sample[['left','stop','right']].values
        y2_cons.append(consensus2.max())

        # load image and mask
        im_name = os.path.join(root_dir,'images',sample.index[0])
        image = krs.load_img(im_name,color_mode='rgb', target_size=imdimensions)
        image = krs.img_to_array(image)
        preproc_im = mn.mobilenet_v2.preprocess_input(image)

        mk_name = os.path.join(root_dir,'masks',sample.index[0]) 
        mask = krs.load_img(im_name,color_mode='grayscale', target_size=imdimensions)
        mask = krs.img_to_array(mask)
        if i==0:
            input1 = np.expand_dims(preproc_im, axis=0)
            y3 = np.expand_dims(mask, axis=0)
            i += 1
        else:
            preproc_im = np.expand_dims(preproc_im, axis=0)
            input1 = np.concatenate((input1,preproc_im))
            mask = np.expand_dims(mask, axis=0)
            y3 = np.concatenate((y3,mask))
        
        # tracking data
        rotfeatures.append([sample.rot_si[0], sample.rot_ap[0], sample.rot_lr[0]])

    csv_features = np.array(rotfeatures)
    input2 = np.concatenate((csv_features, abs(csv_features)), axis=-1)

    inputs = [input1,input2]
    labels = [y1,y2,y3]
    consensus = [y1_cons,y2_cons]

    return inputs, labels, consensus


        
def mobnet_sample(dataframe, root_dir, imdimensions):
    # discard first 9 slices from each sequence
    dataframe['slice'] = dataframe.index.str[-7:-4].astype(int)
    dataframe[dataframe['slice'] >=10]
    #collect a random sample from each class
    p1 = dataframe.loc[dataframe.position.eq('outside')].sample()
    p2 = dataframe.loc[dataframe.position.eq('periphery')].sample()
    p3 = dataframe.loc[dataframe.position.eq('centre')].sample()
    d1 = dataframe.loc[dataframe.direction.eq('left')].sample()
    d2 = dataframe.loc[dataframe.direction.eq('stop')].sample()
    d3 = dataframe.loc[dataframe.direction.eq('right')].sample()
    lst = [p1,p2,p3,d1,d2,d3]
    
    #load sample inputs and masks
    y1=[]
    y1_cons=[]
    y2=[]
    y2_cons=[]
    i=0
    for sample in lst:
        rotfeatures=[]
        # record consensus labels
        y1.append(sample.position[0])
        y2.append(sample.direction[0])

        # record consenus
        consensus1=sample[['outside','periphery','centre']].values
        y1_cons.append(consensus1.max())
        consensus2=sample[['left','stop','right']].values
        y2_cons.append(consensus2.max())

        # load image sequence, tracking data and mask
        for n in range(10):
            slice_name = str(sample.slice[0]-9+n).zfill(3)
            slice_name = sample.index[0][:-7] + slice_name + sample.index[0][-4:]
            im_name = os.path.join(root_dir,'images',sample.index[0])
            image = krs.load_img(im_name,color_mode='rgb', target_size=imdimensions)
            image = krs.img_to_array(image)
            preproc_im = mn.mobilenet_v2.preprocess_input(image)
            if n==0:
                image1 = np.expand_dims(preproc_im, axis=0)
            else:
                preproc_im = np.expand_dims(preproc_im, axis=0)
                image1 = np.concatenate((image1,preproc_im))
            
            # tracking data
            csv_row = dataframe[dataframe.index == slice_name]
            rotfeatures.append([csv_row.rot_si[0], csv_row.rot_ap[0], csv_row.rot_lr[0]])

        csv_features = np.array(rotfeatures)
        rsz_features = np.concatenate((csv_features,abs(csv_features)), axis=-1)
        
        msk_name = os.path.join(root_dir,'masks',sample.index[0]) 
        mask = krs.load_img(im_name,color_mode='grayscale', target_size=imdimensions)
        mask = krs.img_to_array(mask)

        if i==0:
            input1 = np.expand_dims(image1,axis=0)
            input2 = np.expand_dims(rsz_features,axis=0)
            y3 = np.expand_dims(mask, axis=0)
            i += 1
        else:
            image1 = np.expand_dims(image1,axis=0)
            rsz_features = np.expand_dims(rsz_features,axis=0)
            input1 = np.concatenate((input1,image1))
            input2 = np.concatenate((input2,rsz_features))
            mask = np.expand_dims(mask, axis=0)
            y3 = np.concatenate((y3,mask))

    inputs = [input1,input2]
    labels = [y1,y2,y3]
    consensus = [y1_cons,y2_cons]

    return inputs, labels, consensus

def plot_unet_sample(model, dataframe, root_dir, imdimensions, log_dir, epoch):
    sub_titles=['Outside','Periphery','Centre','Left','Stop','Right']
    dpi=300
    im_w = round(imdimensions[1]*6/dpi,1)
    im_h = round(imdimensions[0]*2/dpi,1)
    # check progress on sub-sample of inputs
    inputs_sub, _, consensus_sub=unet_sample(dataframe, root_dir, imdimensions)
    rslts = model.predict(inputs_sub)
    fig=plt.figure(figsize=(im_w,im_h), dpi=300)
    grid = ImageGrid(fig,111,nrows_ncols=(2,6),axes_pad=0)
    for i in range(0,6):
        imdisp = inputs_sub[0][i,...]
        imdisp = imdisp-imdisp.min()
        imdisp = imdisp / imdisp.max()
        grid[i].imshow(imdisp,cmap='gray')
        grid[i].axis('off')
        plt.rcParams.update({'axes.titlesize': 'small'})
        grid[i].set_title(sub_titles[i]+'\nP:'+str(round(consensus_sub[0][i],2))+' D:'+str(round(consensus_sub[1][i],2))+' ')
        imdisp = rslts[0][i,...,2]
        imdisp = imdisp-imdisp.min()
        imdisp = imdisp / imdisp.max()
        grid[i+6].imshow(imdisp,cmap='gray')
        grid[i+6].axis('off')
    plt.savefig(os.path.join(log_dir,'epoch'+str(epoch).zfill(3)+'.jpg'))
    #plt.show(block=True)

def plot_mobnet_sample(model, dataframe, root_dir, imdimensions, log_dir, epoch):
    sub_titles=['Outside','Periphery','Centre','Left','Stop','Right']
    dpi=300
    im_w = round(imdimensions[1]*6/dpi,1)
    im_h = round(imdimensions[0]*2/dpi,1)
    # check progress on sub-sample of inputs
    inputs_sub, _, consensus_sub=mobnet_sample(dataframe, root_dir, imdimensions)
    rslts = model.predict(inputs_sub)
    fig=plt.figure(figsize=(im_w,im_h), dpi=300)
    grid = ImageGrid(fig,111,nrows_ncols=(2,6),axes_pad=0)
    for i in range(0,6):
        imdisp = inputs_sub[0][i,9,...]
        imdisp = imdisp-imdisp.min()
        imdisp = imdisp / imdisp.max()
        grid[i].imshow(imdisp,cmap='gray')
        grid[i].axis('off')
        plt.rcParams.update({'axes.titlesize': 'small'})
        grid[i].set_title(sub_titles[i]+'\nP:'+str(round(consensus_sub[0][i],2))+' D:'+str(round(consensus_sub[1][i],2))+' ')
        imdisp = rslts[2][i,...]
        imdisp = imdisp-imdisp.min()
        imdisp = imdisp / imdisp.max()
        grid[i+6].imshow(imdisp,cmap='gray')
        grid[i+6].axis('off')
    plt.savefig(os.path.join(log_dir,'epoch'+str(epoch).zfill(3)+'.jpg'))
    #plt.show(block=True)