#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Dandan Shan
@Descripation: download all youtube video using lock 
@Date: 2019-04-02 11:57:23
@LastEditTime: 2019-04-05 00:23:57

TODO:
0 merge all
1 remove repeated video_id
2 make sure video exits
3 video_id --> url
4 make (video_id, feature) correspondingly
5 use multi-level folders to save files
'''

import os
import cv2
import json
import argparse
import numpy as np
from skimage import io
from urllib.request import urlretrieve, urlopen
from feature_from_alexnet import get_a_video_feature, get_alexnet_model


def isLocked(filename):
    if os.path.exists(filename):
        return True
    try:
        os.mkdir(filename)
        return False
    except:
        return True


def unLock(filename):
    os.rmdir(filename)


def check_ytvideo_exist(video_id):
    '''
    @description: if all 4 thumbnails exits, then count the video in
    @reutrn: 0 : ignore ; numpy array : features
    '''
    url0 = f'https://img.youtube.com/vi/{video_id}/0.jpg'
    url1 = f'https://img.youtube.com/vi/{video_id}/1.jpg'
    url2 = f'https://img.youtube.com/vi/{video_id}/2.jpg'
    url3 = f'https://img.youtube.com/vi/{video_id}/3.jpg'
    imgs_np = []
    
    try:
        im0 = io.imread(url0)
        im1 = io.imread(url1)
        im2 = io.imread(url2)
        im3 = io.imread(url3)
    
        imgs_list = [im0, im1, im2, im3]
    except:
        #print(f'{video_id} has read thumbnail error !')
        return []
        
    #imgs_np = np.asarray(imgs_list)
    #print(f'{video_id} has read thumbnail successfully !')
    return imgs_list


def get_all_videos_feature(model, all_video_ids):
    #model = model_alexnet
    all_video_feature = []
    exist_video_id = []

    for video_id in all_video_ids:
        imgs_list = check_ytvideo_exist(video_id)
        if len(imgs_list) != 0:
            #print(f'{video_id} get video geature ...')
            one_video_feature_np = get_a_video_feature(model, video_id, imgs_list)
            all_video_feature.append(one_video_feature_np)
            exist_video_id.append(video_id)
    all_video_feature_np = np.asarray(all_video_feature)
    return all_video_feature_np, exist_video_id


def save_video_id_txt(file_path_name, video_id_list):
    with open(file_path_name, "w") as f:
        for item in video_id_list:
            f.write(item + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input folder_path and output_path to run the script !')
    parser.add_argument('--folder_path', default="./vids")
    parser.add_argument('--output_path', default="./output_feature")
    args = parser.parse_args()
    folder_path = args.folder_path
    output_path = args.output_path
    # get model
    model_alexnet = get_alexnet_model()
    # video id list of a genre
    #video_id_list = merge_and_remove_dupilicates(folder_path)
    vidfile_txt_list = os.listdir(folder_path)
    #
    for vidfile_txt in vidfile_txt_list:
        vidfile_txt_path = os.path.join(folder_path, vidfile_txt)
        video_id_list = []
        with open(vidfile_txt_path, "r") as f:
            for item in f:
                item = item.strip()
                video_id_list.append(item)
        print(vidfile_txt + " , id len = " + str(len(video_id_list)))
        
        # use lock to download
        perBlock = 1000
        block_num = len(video_id_list) // 1000 + 1
        genre = vidfile_txt.split("_")[0]
        #
        genre_path = os.path.join(output_path, genre)
        if not os.path.exists(genre_path):
                os.mkdir(genre_path)
        #
        for block_ind in range(block_num):
            ind_range = range(block_ind * perBlock, min(len(video_id_list), (block_ind + 1) * perBlock))

            # subfolder
            subfolder_name = str(block_ind // 20)
            subpath = os.path.join(genre_path, subfolder_name)

            if not os.path.exists(subpath):
                os.mkdir(subpath)
            block_feature_filepath = f'{subpath}/{genre}_feature_block_{block_ind:04d}.npy'
            txt_filepath = f'{subpath}/{genre}_videoid_block_{block_ind:04d}.txt'

            if os.path.exists(block_feature_filepath) or isLocked(block_feature_filepath+".lock"):
                continue
            
            # extract featrure
            block_video_id = video_id_list[ind_range[0]:ind_range[-1]]
            all_video_feature_np, exist_video_id = get_all_videos_feature(model_alexnet, block_video_id)

            # save feature(.npy) and corresponding video_id(.txt)
            np.save(block_feature_filepath, all_video_feature_np)
            save_video_id_txt(txt_filepath, exist_video_id)

            #
            print("*" *30)
            print("genre = "+ genre)
            print("block = " + str(block_ind))
            print("txt len = ", len(exist_video_id))
            print("npy shape = ", all_video_feature_np.shape)
            print("*" *30)

            # unlock the block
            unLock(block_feature_filepath + ".lock")
