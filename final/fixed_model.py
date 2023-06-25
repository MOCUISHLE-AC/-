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
from pycorrector import Corrector
import os
import copy
from flask import Flask, request
#import _init_path
from models.conv import GatedConv
import sys
import json
from ctcdecode import CTCBeamDecoder
from decoder import GreedyDecoder


print("Loading model...")



#######加载语言模型
pwd_path = os.path.abspath(os.path.dirname(__file__))
lm_path = os.path.join(pwd_path, 'lm/zh_giga.no_cna_cmn.prune01244.klm')
language_model = Corrector(language_model_path=lm_path)
alpha = 0.8
beta = 0.3
lm_path = "lm/zh_giga.no_cna_cmn.prune01244.klm"
cutoff_top_n = 40
cutoff_prob = 1.0
beam_width = 32
num_processes = 4
blank_index = 0

def test(
    model,
    epochs=20,
    batch_size=1,
    train_index_path="data/data_aishell/train-sort.manifest",
    dev_index_path="data/data_aishell/test.manifest",
    labels_path="data/data_aishell/labels.json",
    learning_rate=0.6,
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0,
):
    
    
    save_path = './log/'

    print("start dataset")
    train_dataset = data.MASRDataset(train_index_path, labels_path)
  
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    print("end dataset")
    print("start dataloader")
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8
    )
    train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
    )
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=8
    )
    print("end dataloader")
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    #val    
    cer = eval(model, dev_dataloader)
    print("CER = {}".format(cer))
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def translate(vocab, out, out_len):
    return "".join([vocab[x] for x in out[0:out_len]])

def eval(model, dataloader):
    model.eval()
    decoder_greedy=GreedyDecoder(dataloader.dataset.labels_str)
    decoder = CTCBeamDecoder(
    dataloader.dataset.labels_str,
    lm_path,
    alpha,
    beta,
    cutoff_top_n,
    cutoff_prob,
    beam_width,
    num_processes,
    blank_index)
    cer = 0
    cer_lm=0
    print("decoding")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            # print("in decoding")
            # print("xshape")
            # print(x.shape)
            # print("x lens")
            # print(x_lens)
            # print("x")
            # print(x)
            # print("yshape")
            # print(y.shape)
            # print("y lens")
            # print(y_lens)
            # print("y")
            # print(y)
            # print('i')
            # print(i)
            x = x.to("cuda")
            outs, out_lens = model(x, x_lens)
            # print("out shape")
            # print(outs.shape)
            # print("outlen")
            # print(out_lens)

            outs = F.softmax(outs, 1) # 按照维度1进行归一化
            # print("outs shape after softmax")
            # print(outs.shape)
            outs = outs.transpose(1, 2)
            # print("outs shape after transpose")
            # print(outs.shape)

            
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset : offset + y_len])
                offset += y_len
            out,score,offset, out_len = decoder.decode(outs, out_lens)
            print(out.shape)
            out_strings=translate(dataloader.dataset.labels_str, out[0][0], out_len[0][0])
            # out, score, offset, out_len = decoder.decode(y, y_len)
            print("out_strings:",out_strings)
            # print("out_strings.shape[0]:",len(out_strings))


            # out_strings_lm=copy.deepcopy(out_strings)
            # #加载语言模型
            # for index_out in range(0,len(out_strings_lm)):
            #     # print("type:",type(out_strings[index_out][0]))
            #     out_strings_lm[index_out][0], detail=language_model.correct(out_strings_lm[index_out][0])
            # # print("After out_strings:",out_strings)
            # # print("After out_strings.shape[0]:",len(out_strings))
            
            y_strings = decoder_greedy.convert_to_strings(ys)
            print("y_strings:",y_strings[0][0])
            # print("y_strings.shape[0]:",len(y_strings))
            # for pred, truth ,pred_lm in zip(out_strings, y_strings,out_strings_lm):
            #     print('pred使用lm和beamsearch的结果')
            #     print(pred)
            #     print('pred_lm仅仅使用语言模型进行纠错')
            #     print(pred_lm)
            #     print('truth')
            #     print(truth)
            #     trans, ref ,trans_lm= pred[0], truth[0], pred_lm[0]
            #     cer += decoder_greedy.cer(trans, ref) / float(len(ref))
            #     cer_lm+=decoder_greedy.cer(trans_lm, ref) / float(len(ref))
            #     print('cer')
            #     print(cer)
            #     print('cer with lm')
            #     print(cer_lm)
            cer += decoder_greedy.cer(out_strings, y_strings[0][0]) / float(len(y_strings[0][0]))
            print('cer')
            print(cer)

        cer /= len(dataloader.dataset)
    model.train()
    return cer


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    
    with open("data/data_aishell/labels.json") as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)

    model = torch.load("pretrained_ctc/model_100.pth")

    model.to("cuda")


    test(model)



    
    # 没有语言模型是0.124
    # 加上语言模型文本纠错是 CER = 0.12584960509976087
    # 加上beamsearch和语言模型CER = 0.07262976112002184
