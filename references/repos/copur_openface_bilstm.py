
# % markdown 0
'''
<a href="https://colab.research.google.com/github/CopurOnur/Engagement_Detection_OpenFace_Bi-LSTM/blob/main/Engagement_Detection_OpenFace_Bi_LSTM_test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
'''

# % markdown 1
'''
# Engagement Detection in E-Learning Environmets

This notebook presents the code for my thesis named "Engagement Detection in E-Learning Environments". Before running the notebook, please change your run time type to "GPU"
'''

# % markdown 2
'''
## Load libraries (Notebook restart required after running the cell bellow)
'''

# %% cell 3
!nvidia-smi
!pip install --quiet torch
!pip install --quiet pytorch-lightning
!pip install scipy>=1.5
! pip install stumpy
! pip install --quiet tsfresh
!pip install --quiet captum
!pip install -U kora

# % markdown 4
'''
### Run this cell for restart
'''

# %% cell 5
import os
os.kill(os.getpid(), 9)

# % markdown 6
'''
The code chunk bellow access my shared google dirve folder and the necessary files. For authentications you need to sign in with your google account but dont worry, it is not mounting to your drive. 
'''

# %% cell 7
from google.colab import auth

auth.authenticate_user()  # must authenticate


'''list all ids of files directly under folder folder_id'''

def folder_list(folder_id):

  from googleapiclient.discovery import build

  gdrive = build('drive', 'v3').files()

  res = gdrive.list(q="'%s' in parents" % folder_id).execute()

  return [f['id'] for f in res['files']]



'''download all files from a gdrive folder to current directory'''

def folder_download(folder_id):

  for fid in folder_list(folder_id):

    !gdown -q --id $fid

link='https://drive.google.com/drive/folders/1nfh-Qj2xUE5F5qRYiLIVq2RvK8UdU5LX?usp=sharing'


folder_id="1nfh-Qj2xUE5F5qRYiLIVq2RvK8UdU5LX"

folder_download(folder_id)

# % markdown 8
'''
## OpenFace
'''

# % markdown 9
'''
First, the video features are extracted through openface. The code chunck bellow downloads OpenFace and converts the input video into csv file. Downloading OpenFace takes quite some time so you can use "content/mrslowack.csv" to run the model. However, if you want to use a different video, you need to download openface and run it.
'''

# %% cell 10
import os
from os.path import exists, join, basename, splitext

################# Need to revert back to CUDA 10.0 ##################
# Thanks to http://aconcaguasci.blogspot.com/2019/12/setting-up-cuda-100-for-mxnet-on-google.html
#Uninstall the current CUDA version
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update

#Download CUDA 10.0
!wget  --no-clobber https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
#install CUDA kit dpkg
!dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
!sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
!apt-get update
!apt-get install cuda-10-0
#Slove libcurand.so.10 error
!wget --no-clobber http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
#-nc, --no-clobber: skip downloads that would download to existing files.
!apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
!apt-get update
####################################################################

git_repo_url = 'https://github.com/TadasBaltrusaitis/OpenFace.git'
project_name = splitext(basename(git_repo_url))[0]
# clone openface
!git clone -q --depth 1 $git_repo_url

# install new CMake becaue of CUDA10
!wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
!tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

# Get newest GCC
!sudo apt-get update
!sudo apt-get install build-essential 
!sudo apt-get install g++-8

# install python dependencies
!pip install -q youtube-dl

# Finally, actually install OpenFace
!cd OpenFace && bash ./download_models.sh && sudo bash ./install.sh

# % markdown 11
'''
 you can set the path of your new video bellow and the output will be saved to '/content/'
'''

# %% cell 12
video = '/content/pretended.mp4'
newpath = '/content/'

# %% cell 13
! ./OpenFace/build/bin/FeatureExtraction -f $video -out_dir $newpath

# % markdown 14
'''
## Import packages
'''

# %% cell 15
import sys
sys.path.insert(1, '/content')
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import os
import pickle

import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler

from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import fftpack

import math
import utils
import itertools
import torchmetrics
accuray = torchmetrics.Accuracy()

import sys
import dataloader
import model_train
import random
from tsfresh.feature_extraction import extract_features, MinimalFCParameters,feature_calculators
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

# %% cell 16
%matplotlib inline
%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 16, 10

# % markdown 17
'''
# Initial Variables
Please define the variables bellow before running the code. the "csv" variable refers to the extracted openface features. "raw_video" should be the name of the input video. "pic_folder" is the name of the folder that the frames with engagement scores will be saved. After that the these frames will be converted to the output video. "out_video" is the name of the output video. 
'''

# %% cell 18
folder = '/content/'
csv = 'pretend.csv'
raw_video = 'pretend.mp4'
pic_folder = "labeled"
out_video = "out.mp4"


# % markdown 19
'''
# DataLoader
the class bellow is the data dataloader for the model. it takes the input csv, divides it into number of sub clips in which an engagement level will be assigned to each clip. We decided to give assign an engagement level for eac 130 frames reprsented by "self.frame_per_clip" variable. The "self.frame_size" variable represents the number of sequences after statistical aggregation for each clip.
'''

# %% cell 20
class OpenFaceDataset(Dataset):
    ''' Load dataset as torch.tensor '''
    def __init__(self, root=folder + csv):
        self.frame_per_clip = 110
        self.csv = pd.read_csv(root)
        self.frame_size =  5
        self.overlap_size = int(self.frame_per_clip/(self.frame_size*5))


        self.gaze_range=[4,10]
        self.head_range = [10,13]
        self.rot_range = [13,16]
        self.aus_range = [-35,-18]
        self.attributes = ["gaze_seg",
        "head_seg",
        "rot_seg",
        "aus_seg"
        ]
        self.functions = ["length",
                          "maximum","minimum","variance"
        #,"mean_change"
        ]

        self.file_list = self.split_video()
        self.all_features = self.get_feature()
        


    def split_video(self):
      clips = []
      limit = self.csv.shape[0]
      step = int(self.csv.shape[0]/self.frame_per_clip)
      clip_idx = np.linspace(0,limit,step+1,dtype=int)
      for i in range(len(clip_idx)-1):
        if i==0:
          seg = self.csv.iloc[clip_idx[i]:clip_idx[i+1],:]
        else:
          seg = self.csv.iloc[clip_idx[i]- self.overlap_size :clip_idx[i+1] - self.overlap_size,:]
        clips.append(seg)
      return clips


    def get_feature(self):
        features = []
        for idx in range(len(self.file_list)):
            # segment video to 10 segments, return features
            file_dir, label = self.file_list[idx], None
            v_data = np.array(file_dir)
            v_data = np.delete(v_data, 0, 0)    # delete table caption
            v_data = v_data.astype(np.float)   # gaze / pose

            # remove nan
            v_data = v_data[~np.isnan(v_data).any(axis=1)]
            #scaler = MinMaxScaler(feature_range=(0,1))
            #v_data = scaler.fit_transform(v_data)
            #print(v_data.shape)

            limit = v_data.shape[0]
            step = self.frame_size
            frame_idx = np.linspace(0,limit,step+1,dtype=int)
            #print("frames ",frame_idx)
            feature = []
            for i in range(len(frame_idx)-1):
              seg = v_data[frame_idx[i]:frame_idx[i+1],:]
              gaze_seg = seg[:,self.gaze_range[0]:self.gaze_range[1]]
              head_seg = seg[:,self.head_range[0]:self.head_range[1]]
              rot_seg = seg[:,self.rot_range[0]:self.rot_range[1]]
              aus_seg = seg[:,self.aus_range[0]:self.aus_range[1]]

              selected_feature=[]
              for att in self.attributes:
                for func in self.functions:
                  method_to_call = getattr(feature_calculators, func)
                  selected_feature.append(np.apply_along_axis(method_to_call,0,locals()[att]))

              feature.append(torch.FloatTensor(np.concatenate(selected_feature)))
            features.append(feature)

        return features


    def __getitem__(self, idx):
        x = self.all_features[idx]

        data = torch.zeros((self.frame_size,len(x[0])))
        for i in range(self.frame_size):
            data[i,:] = x[i]
        
        return dict(
          sequence = data, #torch.reshape(data,(44,self.frame_size)),
          label = 1)


    def __len__(self):
        return len(self.file_list)

    def get_labels(self):
  
      return self.label_list



# %% cell 21

class OpenFaceDataModule(pl.LightningDataModule):
  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_dataset = OpenFaceDataset()
    self.test_dataset = OpenFaceDataset()

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        #sampler=ImbalancedDatasetSampler(self.train_dataset),
        shuffle=True,
        num_workers=cpu_count()
    )
  
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
  
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
  
    

# %% cell 22
BATCH_SIZE = 8
data_module = OpenFaceDataModule(BATCH_SIZE)
data_module.setup()

# % markdown 23
'''
# Model
Bellow, you can see the code chunk for the Bi-LSTM model.
'''

# %% cell 24
class SequenceModel(nn.Module):
  def __init__(self, n_features, n_hidden=512, n_layers=2,dropout=0.3, freeze_lstm = False):
    super().__init__()
    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.dropout = dropout
    self.freeze_lstm = freeze_lstm

    def weight_init(m):
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


    self.cnn1d = nn.Sequential(
        nn.Conv1d(44,16,3,padding_mode="replicate"),
        nn.ReLU(),
        nn.Conv1d(16,8,3,padding_mode="replicate"),
        nn.ReLU()
    )
    
    self.mlp = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.n_hidden*2 , 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32,1)
    )
    #self.mlp.apply(weight_init)
    self.lstm = nn.LSTM(
         input_size= n_features,
         hidden_size=self.n_hidden,
         num_layers=self.n_layers,
         batch_first=True,
         dropout = self.dropout,
         bidirectional = True
         )
    if freeze_lstm:
      for param in self.lstm.parameters():
        param.requires_grad = False


  def forward(self, x):
    #xr = torch.reshape(self.cnn1d(x),(-1,12,8))
    xr=x
    h0 = torch.zeros(self.n_layers*2, xr.size(0), self.n_hidden)
    h0= h0.type_as(x)
    c0 = torch.zeros(self.n_layers*2, xr.size(0), self.n_hidden)
    c0= c0.type_as(x)
    out,_ = self.lstm(xr,(h0,c0))
    out= out.type_as(x)
    out = self.mlp(out[:,-1, :])
    return out

  
  


# %% cell 25
class EngagementPredictor(pl.LightningModule):

  def __init__(self, n_features: int):
    super().__init__()
    self.model=SequenceModel(n_features)
    self.criterion = nn.MSELoss()

  def forward(self, x, labels=None):
    output=self.model(x)
    loss=0
    if labels is not None:
      loss=self.criterion(output,labels.unsqueeze(dim=1))
      return loss, output
    else:
      return output
      
  
  def training_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self.forward(sequences, labels)
    self.log("train_loss",loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self.forward(sequences, labels)
    self.log("validation_loss",loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self.forward(sequences, labels)
    self.log("test_loss",loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return optim.AdamW(self.parameters(), lr=0.0001)

# %% cell 26
model = EngagementPredictor(
    n_features = data_module.train_dataset[0]["sequence"].shape[1])

# % markdown 27
'''
# Load the pre-trained weights
'''

# %% cell 28
trained_model = EngagementPredictor.load_from_checkpoint(
    "/content/trained_model_weights.ckpt",
    n_features = data_module.train_dataset[0]["sequence"].shape[1],
    n_hidden = 512,
    n_layers = 2,
    dropout = 0.3)

trained_model.freeze()


# %% cell 29
labels = []
predictions = []

for item in tqdm(data_module.val_dataloader()):
  sequence = item["sequence"]
  label = item["label"]
  _, output = trained_model(sequence,label)
  predictions.append(output)
  labels.append(label.item())

# % markdown 30
'''
# Put Engagement Labels on the video frames
'''

# %% cell 31
import cv2
from google.colab.patches import cv2_imshow
cap = cv2.VideoCapture(folder + raw_video)
count = 0
limit = data_module.train_dataset.csv.shape[0]
step = len(predictions)
tresh = np.linspace(0,limit,step+1,dtype=int)
while(True):
      
    # Capture frames in the video
    ret, frame = cap.read()
  
    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # Use putText() method for
    # inserting text on video
    for i in range(1,step+1):
      if count<tresh[i]:
        acc = predictions[i-1]
        count+=1
        break

    cv2.putText(frame, 
                'Engagement Level is '+ str(acc.item()), 
                (25, 25), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

    from pathlib import Path
    Path(folder + pic_folder).mkdir(parents=True, exist_ok=True)
    # Display the resulting frame
    try:
      cv2.imwrite(folder + pic_folder+ "/frame%d.jpg" % count, frame)
    except:
      break
  
# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()

# % markdown 32
'''
# Convert the labeled frames to video
'''

# %% cell 33
import glob
import natsort
sorted = natsort.natsorted(os.listdir(folder + pic_folder),reverse=False)
img_array = []
for filename in sorted:
    img = cv2.imread(folder + pic_folder + "/" + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# %% cell 34
!ffmpeg -i project.avi output.mp4

# % markdown 35
'''
# Display the Video
'''

# %% cell 36
from kora.drive import upload_public
url = upload_public('/content/output.mp4')
# then display it
from IPython.display import HTML
HTML(f"""<video src={url} width=500 controls/>""")
