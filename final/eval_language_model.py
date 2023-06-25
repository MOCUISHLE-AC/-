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
#######加载语言模型
pwd_path = os.path.abspath(os.path.dirname(__file__))
lm_path = os.path.join(pwd_path, 'lm/zh_giga.no_cna_cmn.prune01244.klm')
language_model = Corrector(language_model_path=lm_path)


def train(
    model,
    epochs=20,
    batch_size=64,
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


def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
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
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            # print("out_strings:",out_strings)
            # print("out_strings.shape[0]:",len(out_strings))
            # #加载语言模型
            for index_out in range(0,len(out_strings)):
                print("type:",type(out_strings[index_out][0]))
                out_strings[index_out][0], detail=language_model.correct(out_strings[index_out][0])
            # print("After out_strings:",out_strings)
            # print("After out_strings.shape[0]:",len(out_strings))
            
            y_strings = decoder.convert_to_strings(ys)
            #print("y_strings:",y_strings)
            # print("y_strings.shape[0]:",len(y_strings))
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder.cer(trans, ref) / float(len(ref))
                # print("in cer cal")
                # print("pred")
                # print(pred)
                # print("truth")
                # print(truth)
                # print("trans")
                # print(trans)
                # print("ref")
                # print(ref)
                # print("decoder.cer(trans, ref)")
                # print(decoder.cer(trans, ref))
                # print("float(len(ref))")
                # print(float(len(ref)))
               

             

                
        cer /= len(dataloader.dataset)
    model.train()
    return cer


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    
    with open("data/data_aishell/labels.json") as f:
        vocabulary = json.load(f)
        vocabulary = "".join(vocabulary)
        # print(vocabulary)
    print("start training 1")
    # torch.backends.cudnn.enabled = False
    model = torch.load("pretrained_ctc/model_100.pth")

    print("start training 2")
   
    model.to("cuda")
    print(torch.cuda.is_available())
    print("start training 3")
    
    x=torch.randn(100)
    y=x.cuda()
    print(type(y))
    print((y))
    train(model)
