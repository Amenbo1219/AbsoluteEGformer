import os
from torch.utils import data
from torchvision import transforms
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
#import OpenEXR, Imath, array
import math
import os.path as osp
import torch.utils.data
from skimage.transform import rescale,resize
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
                self.semantic_paths = []
                dir_sub_dir=[]

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'2D_rendering')
                    sub_path_list = os.listdir(sub_path)
                    

                    for path in sub_path_list:
                        dir_sub_dir.append(os.path.join(sub_path,path,'panorama/full'))
                

                for final_path in dir_sub_dir:
                    self.image_paths.append(os.path.join(final_path,'rgb_rawlight.png'))                    
                    self.depth_paths.append(os.path.join(final_path,'depth.png'))                    
                    self.semantic_paths.append(os.path.join(final_path,'semantic.png'))                    
                # self.image_paths + self.image_paths[:1000]    
                # self.depth_paths + self.depth_paths[:1000]    
                print("ImageNums;",self.__len__())
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
            space,ns = self.get_sparse_depth(depth,num_sample=500)

            data=[]

        # if self.transform is not None:
            data = {'color':image,'depth':depth, 'mask':mask,'space':space}
            
        return data
    
    def get_sparse_depth(self, dep, num_sample, test=False, max_=500):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)

        if test:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.args.sample_seed)
            idx_sample = torch.randperm(num_idx, generator=g_cpu)[:num_sample]
        else:
            if num_sample == 'random' or num_sample=='random_high500':
                if num_sample == 'random_high500':
                    if random.randint(1, 2) == 2:
                        num_sample = 500
                    else:
                        num_sample = random.randint(1, 500)
                else:
                    num_sample = random.randint(1, max_)
                
            else:
                num_sample = int(num_sample)
            idx_sample = torch.randperm(num_idx)[:num_sample]   
        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)
        return dep_sp, num_sample        
        
    def __len__(self):
        
        return len(self.image_paths)





class Sample_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None):
            "makes directory list which lies in the root directory"
            if True:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                self.image_paths = []
                self.depth_paths = []
                dir_sub_dir=[]


                index = 0 
                for path in dir_path:
                    file_list = os.listdir(path)
                    for file_name in file_list:
                        self.image_paths.append(os.path.join(path,file_name))

                        index = index + 1

                self.image_paths.sort()
                self.transform = transform
                self.transform_t = transform_t

    def __getitem__(self,index):
           
        if True:
            
            image_path = self.image_paths[index]
                
            image = Image.open(image_path).convert('RGB')
            # image = image.resize((768,384))


            data=[]

        if self.transform is not None:
            data.append(self.transform(image))
        return data

    def __len__(self):
        
        return len(self.image_paths)


