import os
import torch
import torch.nn as nn
import data
from models.conv import GatedConv
from tqdm import tqdm
from decoder import GreedyDecoder
from torch.nn import CTCLoss
import tensorboardX as tensorboard
import torch.nn.functional as F
import json
# from pycorrector import Corrector
import os
#######加载语言模型
# pwd_path = os.path.abspath(os.path.dirname(__file__))
# lm_path = os.path.join(pwd_path, 'lm/zh_giga.no_cna_cmn.prune01244.klm')
# language_model = Corrector(language_model_path=lm_path)



if __name__ == "__main__":
    model = torch.load("pretrained_ctc/model_attention_100.pth")
    torch.save(model.state_dict(), "pretrained_ctc/dict.pt")  # 保存