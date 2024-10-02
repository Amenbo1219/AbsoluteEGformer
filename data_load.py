import os
from torch.utils import data
from torchvision import transforms
import math
from PIL import ImageEnhance
import random
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import RandomSampler
import torchvision.transforms.functional as F
from imageio import imread
import numpy as np
from skimage import io
import math
import os.path as osp
import torch.utils.data

class S3D_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None):
            "makes directory list which lies in the root directory"
            if True:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.image_paths = []
                self.depth_paths = []
                dir_sub_dir=[]
 

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'2D_rendering')
                    sub_path_list = os.listdir(sub_path)
                    

                    for path in sub_path_list:
                        dir_sub_dir.append(os.path.join(sub_path,path,'panorama/full'))
                

                for final_path in dir_sub_dir:
                    self.image_paths.append(os.path.join(final_path,'rgb_rawlight.png'))                    
                    self.depth_paths.append(os.path.join(final_path,'depth.png'))                    



                self.transform = transform
                self.transform_t = transform_t

    def __getitem__(self,index):
           
        if True:
            
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
            import cv2
            image = np.array(Image.open(image_path).convert('RGB'),np.float32)
            # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32).reshape(image.shape[0],image.shape[1],1)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32).reshape(image.shape[0],image.shape[1],1)
            mask = (depth>0.)
            image = (torch.from_numpy((image/(pow(2,8)-1)).transpose(2,0,1)))
            depth = (torch.from_numpy((depth/1000.).transpose(2,0,1)))
            mask = (torch.from_numpy((mask/1000.).transpose(2,0,1)))
            data=[]

        # if self.transform is not None:
            data = {'color':image,'depth':depth, 'mask':mask}            
        return data
    def __len__(self):
        
        return len(self.image_paths)
