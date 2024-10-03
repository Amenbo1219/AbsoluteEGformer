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
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
import importlib
import numpy as np
import cv2


class S3D_loader():
    def __init__(self,root):
            "makes directory list which lies in the root directory"
            self.gt_paths =[]
            self.ngt_paths =[]
            self.relative_paths = []
            self.space_paths = []
            self.nspace_paths = []
            self.init_paths = []
            self.ninit_paths = []
            self.rgb_paths = []
            if True:
                
                files = list(os.listdir(root))
                numeric_files = []
                count = 0
                for file in files:
                    name, ext = os.path.splitext(file)
                    if name[:-3].isdigit():
                        # numeric_files.append(name[:-3])
                        numeric_files.append(str(count))
                        count += 1
                print(numeric_files)
                for file in numeric_files :
                    fpath = os.path.join(root,file)
                    # self.gt_paths.append(os.path.join(fpath+'_gt.png'))  # GT 用の分岐
                    self.gt_paths.append(os.path.join(fpath+'_gt.png'))  # GT 用の分岐
                    self.ngt_paths.append(os.path.join(fpath+'_ngt.png'))
                    # print(self.gt_paths)

                    self.init_paths.append(os.path.join(fpath+'_init.png'))
                    self.ninit_paths.append(os.path.join(fpath+'_ninit.png'))
                    self.relative_paths.append(os.path.join(fpath+'_rel.png'))  # rel 用の分岐
                    self.space_paths.append(os.path.join(fpath+'_space.png'))  # space 用の分岐
                    self.rgb_paths.append(os.path.join(fpath+'_rgb.png'))  # rgb 用の分岐
                    
                

    def get(self,index):
           
        if True:
            import cv2
            try :
                gt_p = self.gt_paths[index]
            except:
                assert False, f"gt_path is not found."
            
            if os.path.isfile(gt_p) == True:
                gt = cv2.imread(gt_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
                # gt = gt.reshape(gt.shape[0],gt.shape[1],1)
            else:    
                print(f"{gt_p} is not found.")

            ngt_p = self.ngt_paths[index]
            if os.path.isfile(ngt_p):
                ngt = cv2.imread(ngt_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            else:
                ngt = None

            init_p = self.init_paths[index]
            if os.path.isfile(init_p):
                init = cv2.imread(init_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            else:
                init = None

            ninit_p = self.ninit_paths[index]
            if os.path.isfile(ninit_p):
                ninit = cv2.imread(ninit_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            else:
                ninit = None    
            
            rel_p = self.relative_paths[index]
            if os.path.isfile(rel_p) :
                rel= cv2.imread(rel_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            else:
                rel = None

            space_p = self.space_paths[index]
            if os.path.isfile(space_p) :
                space = cv2.imread(space_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            else:
                space = None
            rgb_p = self.rgb_paths[index]
            if os.path.isfile(rgb_p) == True:
                image = np.array(Image.open(rgb_p).convert('RGB'),np.uint8)
            else:
                image = None
            

            
            data = {'gt':gt,'ngt':ngt,'init':init,'ninit':ninit,'rgb':image,'rel':rel,"space":space}
        return data

    

    def __len__(self):
        
        return len(self.gt_paths)
if __name__ == "__main__":
    root = "./output"
    output_path = "./rgb_output"
    del_loader = S3D_loader(root)
    del_loader.get
    n_max = 10
    masic_no = 1 # 必要があれば変更して
    psnr = []
    for cnt in range(del_loader.__len__()):
        data = del_loader.get(cnt)
        # print(data)
        tmp = []
        for key in data.keys():
            # print(key)
            img = data[key]
            img*=masic_no
            if key=="gt":
            # if key!="rgb":
                n_max = img.max()
                # print(n_max)
                vmax = np.percentile(img/n_max, 95)
                normalizer = mpl.colors.Normalize(vmin=(img/n_max).min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
            if img is not None:
                if key == 'rgb':
                    cv2.imwrite(os.path.join(output_path,f"{cnt}_{key}_vis.png"),img.astype(np.uint8))
                    continue
                img = img.astype(np.float32)
                n_max = img.max()
                img /= n_max
                # img = np.clip((img*255),0,255)
                img = mapper.to_rgba(img)
                if key == "gt" or key == "ngt":
                    img_tmp = np.clip((img*(pow(2, 16)-1)),0,pow(2, 16)-1).astype(np.uint16)
                    tmp.append(img_tmp)
                img = np.clip((img*255),0,255).astype(np.uint8)
                # img = mapper.to_rgba(img)
                # img = np.clip((img*pow(2, 16)), 0, pow(2, 16)).astype(np.uint16)
                print(img.shape,img.min(),img.max(),img.mean())
                plt.imsave(os.path.join(output_path,f"{cnt}_{key}_vis.png"),img)
                
        psnr.append(cv2.PSNR(tmp[0], tmp[1], pow(2, 16)-1))
    psnr_np = np.array(psnr)    
    print("[psnr] shape:{}, max:{}, min:{}, mean:{}".format(psnr_np.shape, psnr_np.max(), psnr_np.min(), psnr_np.mean()))
