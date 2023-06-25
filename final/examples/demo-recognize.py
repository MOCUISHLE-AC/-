import _init_path
import torch
from models.conv import GatedConv
import os
import torch.nn as nn
import data
from tqdm import tqdm
from decoder import GreedyDecoder
from torch.nn import CTCLoss
import tensorboardX as tensorboard
import torch.nn.functional as F
import json


#pretrained/gated-conv.pth
#pretrained_ctc/model_19.pth
if __name__ == "__main__":
    model = torch.load("pretrained_ctc/model_19.pth")
    model.to("cpu")

    text = model.predict("test.wav")

    print("")
    print("识别结果:")
    print(text)
