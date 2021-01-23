""" Data loader for  LTSM-net """

import datetime
import pandas as pd
from sklearn.utils import class_weight
import os
from tensorflow.keras.preprocessing import image as krs
from tensorflow.keras import applications as mn
from PIL import Image , ImageOps
import random
import numpy as np

# write user-defined parameters to file
#with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
#    print(args, file=f)
#with open(os.path.join(log_dir, 'params.txt'), 'a') as f:
#    print(os.path.basename(sys.argv[0]), file=f)


def us_generator(dataframe, img_path, msk_path, batch_size, imdimensions=(512,512)):
    # for each sample, list image name and calculate class weights
    images_list = dataframe.index.tolist()
    #pths_list = dataframe.pths.tolist()
    pos_list = dataframe.position.tolist()
    dir_list = dataframe.direction.tolist()
    pos_weight = class_weight.compute_sample_weight('balanced', pos_list)
    dir_weight = class_weight.compute_sample_weight('balanced', dir_list)
    # perform initial shuffle
    random.Random(30).shuffle(images_list)
    random.Random(30).shuffle(pos_weight)
    random.Random(30).shuffle(dir_weight)
    i = 0
    while True:
        # preallocate output for this batch
        batch_x = {'imagesS': [], 'rotsS': []}
        batch_y = {'prostate_out': [], 'direction_out': [], 'segment_out': []}
        batch_w = {'prostate_out': [], 'direction_out': []}
        # loop through each sample (b) assigned to this batch
        b = 0
        while b < batch_size:
            # refresh shuffle
            if i == len(images_list):
                i = 0
                random.Random(42).shuffle(images_list)
                random.Random(42).shuffle(pos_weight)
                random.Random(42).shuffle(dir_weight)

            if int(images_list[i][-7:-4]) > 9:
                filelist = [os.path.join(img_path, images_list[i][:-7]+str(int(images_list[i][-7:-4])-9+n).zfill(3) + images_list[i][-4:]) for n in range(10)]

                masklist = [os.path.join(msk_path, 'msk_'+images_list[i][:-7]+str(int(images_list[i][-7:-4])-9+n).zfill(3) + images_list[i][-4:]) for n in range(10)]

                ixlist = [images_list[i][:-7]+str(int(images_list[i][-7:-4])-9+n).zfill(3) + images_list[i][-4:] for n in range(10)]
                
                # augmentation parameters
                rnd_angle = random.randint(-25, 25)
                #rnd_lr = random.randint(0, 1)
                #rnd_ud = random.randint(0, 1)
                rnd_x = random.randint(-100, -100)
                rnd_y = random.randint(-100, 100)

                # preassign probe angle vector list
                rotfeatures=[]

                # load 5 images in sequence
                for n in range(10):
                    # augment images
                    image1 = krs.load_img(filelist[n],color_mode='rgb', target_size=imdimensions)
                    image1.rotate(rnd_angle)
                    image1.transform(image1.size, Image.AFFINE,(1,0,rnd_x,0,1,rnd_y))
                    # augment masks
                    if n==9:
                        mask1 = krs.load_img(masklist[n],color_mode='grayscale', target_size=imdimensions)
                        mask1.rotate(rnd_angle)
                        mask1.transform(mask1.size, Image.AFFINE,(1,0,rnd_x,0,1,rnd_y))
                        # ensure correct range and scaling
                        mask1 = krs.img_to_array(mask1)
                        mask1 = np.clip(mask1,0,1)
                        mask=mask1+1

                    image1 = krs.img_to_array(image1)
                    image1 = mn.mobilenet_v2.preprocess_input(image1)  # ensure scaling is appropriate to model
                    if n == 0:
                        image = np.expand_dims(image1, axis=0)
                    else:
                        image1 = np.expand_dims(image1, axis=0)
                        image = np.concatenate((image, image1))

                    # assign probe angle vector to sequence
                    csv_row = dataframe.loc[ixlist[n], :]
                    rotfeatures.append([csv_row['rot_si'], csv_row['rot_ap'], csv_row['rot_lr']])
                    
                # embed probe vectors into array of equal size to images (necessary for TimeDistributed model wrapper)
                csv_features = np.array(rotfeatures)
                rsz_features = np.concatenate((csv_features, abs(csv_features)), axis=-1)

                # record sequence-level labels from csv
                csv_row = dataframe.loc[images_list[i], :]
                labelPos = np.array([csv_row['outside'], csv_row['periphery'], csv_row['centre']])
                labelDir = np.array([csv_row['left'], csv_row['stop'], csv_row['right']])
                # record class weights to balance class sizes
                wt_pos = pos_weight[i]
                wt_dir = dir_weight[i]

                # append each record to the batch
                batch_x['imagesS'].append(image)
                batch_x['rotsS'].append(rsz_features)
                batch_y['prostate_out'].append(labelPos)
                batch_y['direction_out'].append(labelDir)
                batch_y['segment_out'].append(mask)
                batch_w['prostate_out'].append(wt_pos)
                batch_w['direction_out'].append(wt_dir)
                i += 1
                b += 1
            else:
                i += 1

        batch_x['imagesS'] = np.array(batch_x['imagesS'])
        batch_x['rotsS'] = np.array(batch_x['rotsS'])
        batch_y['prostate_out'] = np.array(batch_y['prostate_out'])
        batch_y['direction_out'] = np.array(batch_y['direction_out'])
        batch_y['segment_out'] = np.array(batch_y['segment_out'])
        batch_w['prostate_out'] = np.array(batch_w['prostate_out'])
        batch_w['direction_out'] = np.array(batch_w['direction_out'])

        yield(batch_x, batch_y, batch_w)


##### non-sequential generator
def us_single(dataframe, img_path, msk_path, batch_size,imdimensions=(640,480)):
    # for each sample, list image name and calculate class weights
    images_list = dataframe.index.tolist()
    #pths_list = dataframe.pths.tolist()
    pos_list = dataframe.position.tolist()
    dir_list = dataframe.direction.tolist()
    pos_weight = class_weight.compute_sample_weight('balanced', pos_list)
    dir_weight = class_weight.compute_sample_weight('balanced', dir_list)
    # perform initial shuffle
    random.Random(30).shuffle(images_list)
    random.Random(30).shuffle(pos_weight)
    random.Random(30).shuffle(dir_weight)
    i = 0
    while True:
        # preallocate output for this batch
        batch_x = {'images': [], 'rots': []}
        batch_y = {'prostate_out': [], 'direction_out': [], 'segment_out': []}
        batch_w = {'prostate_out': [], 'direction_out': []}
        # loop through each sample (b) assigned to this batch
        b = 0
        while b < batch_size:
            # refresh shuffle
            if i == len(images_list):
                i = 0
                random.Random(42).shuffle(images_list)
                random.Random(42).shuffle(pos_weight)
                random.Random(42).shuffle(dir_weight)

            filelist = os.path.join(img_path, images_list[i])
            masklist = os.path.join(msk_path, 'msk_'+images_list[i])
            ixlist = images_list[i]
            
            # augmentation parameters
            rnd_angle = random.randint(-25, 25)
            #rnd_lr = random.randint(0, 1)
            #rnd_ud = random.randint(0, 1)
            rnd_x = random.randint(-100, -100)
            rnd_y = random.randint(-100, 100)

            # augment images
            image1 = (krs.load_img(filelist,color_mode='rgb', target_size=imdimensions))
            image1.rotate(rnd_angle)
            image1.transform(image1.size, Image.AFFINE,(1,0,rnd_x,0,1,rnd_y))
            # augment masks
            mask1 = (krs.load_img(masklist,color_mode='grayscale', target_size=imdimensions))
            mask1.rotate(rnd_angle)
            mask1.transform(mask1.size, Image.AFFINE,(1,0,rnd_x,0,1,rnd_y))
            # ensure correct range and scaling
            mask1 = krs.img_to_array(mask1)
            mask1 = np.clip(mask1,0,1)
            mask=mask1+1
            image1 = krs.img_to_array(image1)
            image = mn.mobilenet_v2.preprocess_input(image1)  # ensure scaling is appropriate to model

            # assign probe angle vector to sequence
            csv_row = dataframe.loc[ixlist, :]                
            # embed probe vectors into array of equal size to images (necessary for TimeDistributed model wrapper)
            csv_features = np.array([csv_row['rot_si'], csv_row['rot_ap'], csv_row['rot_lr']])
            rsz_features = np.concatenate((csv_features, abs(csv_features)), axis=-1)

            # record sequence-level labels from csv
            labelPos = np.array([csv_row['outside'], csv_row['periphery'], csv_row['centre']])
            labelDir = np.array([csv_row['left'], csv_row['stop'], csv_row['right']])
            # record class weights to balance class sizes
            wt_pos = pos_weight[i]
            wt_dir = dir_weight[i]

            # append each record to the batch
            batch_x['images'].append(image)
            batch_x['rots'].append(rsz_features)
            batch_y['prostate_out'].append(labelPos)
            batch_y['direction_out'].append(labelDir)
            batch_y['segment_out'].append(mask)
            batch_w['prostate_out'].append(wt_pos)
            batch_w['direction_out'].append(wt_dir)
            i += 1
            b += 1
        else:
            i += 1

        batch_x['images'] = np.array(batch_x['images'])
        batch_x['rots'] = np.array(batch_x['rots'])
        batch_y['prostate_out'] = np.array(batch_y['prostate_out'])
        batch_y['direction_out'] = np.array(batch_y['direction_out'])
        batch_y['segment_out'] = np.array(batch_y['segment_out'])
        batch_w['prostate_out'] = np.array(batch_w['prostate_out'])
        batch_w['direction_out'] = np.array(batch_w['direction_out'])

        yield(batch_x, batch_y, batch_w)
        