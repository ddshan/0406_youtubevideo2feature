import os
import copy
import pickle
import numpy as np
from PIL import Image
import urllib.request
from skimage import io
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.externals import joblib
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable 

            
def get_alexnet_model(feature_extract=True, use_pretrained=True):
    '''
    @description: get alexnet[:13]
    '''
    model = models.alexnet(pretrained=True)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    alexnet_pool_model = model.features[:13]

    return alexnet_pool_model

def get_a_video_feature(model, video_id, imgs_4np, use_cuda=True):

    feature_4thumbails = []
    model = model.cuda()

    for img in imgs_4np:

        img = Image.fromarray(img)
        img = img.crop(img.getbbox())
        img = img.resize((227, 227))
        img_tensor = transforms.ToTensor()(img)
        #img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.resize_(1,3,227,227)
        
        if use_cuda :
            img_tensor = img_tensor.cuda()
        feature_tensor = torch.squeeze(model(Variable(img_tensor)))

        # process feature
        result = feature_tensor.view(256, -1) #[256,36]
        result = F.normalize(result, dim=1)  #normalization
        result = torch.mean(result, dim=1, keepdim=True).view(1,-1)
        feature_np = result.cpu().numpy()
        #
        feature_4thumbails.append(feature_np)
    #
    # avg and distance
    Dis = np.zeros((1, 6)) # distance

    count = 0
    for m in range(3):
        for n in range(m + 1, 4):
            Dis[0, count] = np.dot(feature_4thumbails[m], feature_4thumbails[n].T)
            count += 1
            
    Dmin = Dis.min()
    Dmean = Dis.mean()
    Dmax = Dis.max()
    Mean = (feature_4thumbails[0]+feature_4thumbails[1]+feature_4thumbails[2]+feature_4thumbails[3])/4
    feature_result = np.append(Mean, [Dmin, Dmean, Dmax])
    #
    return np.asarray(feature_result)



# def prepare_feature_and_score(phase, txtfile_path, folder_path, model_avg, model_fc, model_name, genre):
#     X_avg_feature = []
#     Y_avg_score = []
    
#     X_fc_feature = []
#     Y_fc_score = []
    
#     id_score_dic = {}
    
#     with open(txtfile_path, "r") as f:
#         for item in f:
#             item = item.strip()
#             video_id, score = item.split(" ")
#             id_score_dic[video_id] = int(score)/100.0
            
#     video_id_list = os.listdir(folder_path)
#     for item in video_id_list:
#         avg_feature, fc_feature = get_feature_4thumbnail(model_avg, model_fc, folder_path, item)
#         X_avg_feature.append(avg_feature)
#         Y_avg_score.append(id_score_dic[item])
#         X_fc_feature.append(fc_feature)
#         Y_fc_score.append(id_score_dic[item])
#     np.save(genre + "/" + phase + "_" + model_name + "_" + genre + "_X_avg_feature", X_avg_feature)
#     np.save(genre + "/" + phase + "_" + model_name + "_" + genre + "_Y_avg_score", Y_avg_score)
#     np.save(genre + "/" + phase + "_" + model_name + "_" + genre + "_X_fc_feature", X_fc_feature)
#     np.save(genre + "/" + phase + "_" + model_name + "_" + genre + "_Y_fc_score", Y_fc_score)
    
#     print(phase + " " + model_name + " " + genre)
#     print("*" * 30)
#     print(np.asarray(X_avg_feature).shape)
#     print(np.asarray(Y_avg_score).shape)
#     print(np.asarray(X_fc_feature).shape)
#     print(np.asarray(Y_fc_score).shape)
    
#     return X_avg_feature, Y_avg_score, X_fc_feature, Y_fc_score, video_id_list

# def build_clf_and_predict(Xtrain, Ytrain, Xtest, Ytest, test_video_id_list, result_output_filepath, clf_model_output_path):
#     #
#     #
#     clf = SVR(gamma='scale', C=1.0, epsilon=0.2, kernel='linear')
#     clf.fit(Xtrain, Ytrain)
#     Ypred = clf.predict(Xtest)
#     print(Ypred)
#     fout = open(result_output_filepath, "w")
#     for i in range(len(test_video_id_list)):
#         vid = test_video_id_list[i]
#         ytestscore = Ytest[i]
#         ypredscore = Ypred[i]
#         fout.write(vid + " " + str(ytestscore) + " " + str(ypredscore) + "\n")      
#     joblib.dump(clf, clf_model_output_path)

#     return Ytest, Ypred


# def get_l1_2_and_cc(Ytest, Ypred):
#     Ytest = np.asarray(Ytest)
#     Ypred = np.asarray(Ypred)
    
#     delta = Ytest - Ypred
#     l1 = np.sum(np.absolute(delta))
#     l2 = np.sum(np.power(delta, 2))
#     test_sorted = sorted(Ytest)
#     pred_sorted = sorted(Ypred)
#     #
#     n = len(Ypred)
#     test_index = [test_sorted.index(v) for v in Ytest]
#     pred_index = [pred_sorted.index(v) for v in Ypred]
    
#     test_index = np.asarray(test_index)
#     pred_index = np.asarray(pred_index)
    
#     Sum_d = np.sum(np.power(pred_index-test_index, 2))
#     cc = 1- 6 * Sum_d / ( n*(n*n-1) )

#     print("l1 = ", l1)
#     print("l2 = ", l2)
#     print("cc = ", cc)



# if __name__=="__main__":
#     model_name = "alexnet"
#     genre = "diy"
#     test_file_path = "testset/diy_test_200_handscore.txt"
#     train_file_path = "trainset/diy_train_1000_handscore.txt"
#     test_folder_path="testset/diy_tn"
#     train_folder_path="trainset/diy_tn"
#     #
#     avg_result_output_filepath = genre + "/" + model_name+"_"+genre+"_avg_result_output.txt"
#     avg_clf_model_output_path =  genre + "/" + model_name+"_"+genre+"_avg_clf_model.joblib"
#     #
#     fc_result_output_filepath = genre + "/" + model_name+"_"+genre+"_fc_result_output.txt"
#     fc_clf_model_output_path = genre + "/" + model_name+"_"+genre+"_fc_clf_model.joblib"
#     #
#     model_avg, model_fc = get_model(model_name)


#     print("********************")
#     print(model_name + " " + genre)
#     print("test_file_path: ",test_file_path)
#     print("train_file_path:", train_file_path)
#     print("test_folder_path: ", test_folder_path)
#     print("train_folder_path: ", train_folder_path)
#     print("********************")
#     # get train data
#     print("get train data ...")
#     train_X_avg_feature, train_Y_avg_score, train_X_fc_feature, train_Y_fc_score, train_video_id_list = prepare_feature_and_score("train", train_file_path, train_folder_path, model_avg, model_fc, model_name, genre)
#     # get test data
#     print("get test data ...")
#     test_X_avg_feature, test_Y_avg_score, test_X_fc_feature, test_Y_fc_score, test_video_id_list = prepare_feature_and_score("test", test_file_path, test_folder_path, model_avg, model_fc, model_name, genre)
#     # avg
#     print("pred avg ...")
#     Ytest_avg, Ypred_avg = build_clf_and_predict(train_X_avg_feature, train_Y_avg_score, test_X_avg_feature, test_Y_avg_score, test_video_id_list, avg_result_output_filepath, avg_clf_model_output_path)
#     get_l1_2_and_cc(Ytest_avg, Ypred_avg)
#     # fc 
#     print("pred fc ...")
#     Ytest_fc, Ypred_fc = build_clf_and_predict(train_X_fc_feature, train_Y_fc_score, test_X_fc_feature, test_Y_fc_score, test_video_id_list, fc_result_output_filepath, fc_clf_model_output_path)
#     get_l1_2_and_cc(Ytest_fc, Ypred_fc)












