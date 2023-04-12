# 语音信息处理实验3 

------

<p align="left">
    姓名：雷贺奥<br> 
    学号：2013551<br> 
    专业：计算机科学与技术<br> 
</p> 

## 实验描述

1.根据提供的语料库和参考代码，构建FNN语言模型，提交运行结果截图。

2.使用tensorboard可视化模型在训练集和验证集上的loss曲线，以每个epoch为单位。

3.使用训练好的语言模型，计算以下两句话的困惑度：“Jane went to the store”和“store to Jane went the”

4.改进模型（可选）

## 实验过程

### 实验代码

#### 源代码

```python
try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))

if not os.path.exists("./logs"):
    os.makedirs("./logs")
writer = SummaryWriter('./logs')

for epoch in range(5):
  # Perform training
  random.shuffle(train)
  # set the model to training mode
  model.train()
  train_words, train_loss = 0, 0.0
  start = time.time()
  print(f'Starting training epoch {epoch+1} over {len(train)} sentences')
  for sent_id, sent in tqdm(enumerate(train)):
    my_loss = calc_sent_loss(sent)
    train_loss += my_loss.data
    train_words += len(sent)
    optimizer.zero_grad()
    my_loss.backward()
    optimizer.step()
    if (sent_id+1) % 5000 == 0:
      print("--finished %r sentences (word/sec=%.2f)" % (sent_id+1, train_words/(time.time()-start)))
  print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (epoch, train_loss/train_words, math.exp(train_loss/train_words), train_words/(time.time()-start)))
  #绘图
  writer.add_scalar("loss/train", train_loss/train_words, epoch)
```

#### 调整网络结构

~~~python
class FNN_LM(nn.Module):
  def __init__(self, nwords, emb_size, hid_size, num_hist,droupout=0.2):
    super(FNN_LM, self).__init__()
    self.embedding = nn.Embedding(nwords, emb_size)
    self.fnn = nn.Sequential(
      # nn.Linear(num_hist*emb_size, hid_size),
      # nn.Tanh(),
      # nn.Linear(hid_size, nwords)
      nn.Linear(num_hist*emb_size, hid_size),
      #防止训练过度
      nn.Dropout(droupout),
      #激活函数
      nn.Tanh(),
      nn.Linear(hid_size, hid_size*4),
      nn.Dropout(droupout),
      nn.Tanh(),
      nn.Linear(hid_size*4, hid_size),
      nn.Dropout(droupout),
      nn.Tanh(),
      nn.Linear(hid_size, nwords)
    )
~~~

#### 困惑度

~~~python
mytest = list(read_dataset("../data/ptb-text/mytest.txt", add_vocab=False))
model.eval()
for send_id, sent in enumerate(mytest):
	loss = calc_sent_loss(sent)
    print("id=%d loss=%f ppl=%f"%(send_id,loss.data, math.exp(loss.data/len(sent))))
~~~

### 实验结果

#### 代码运行结果：

![image-20230407104645429](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230407104645429.png)

![image-20230407104816777](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230407104816777.png)

#### loss图像

**源代码$epoch=5$:**

![image-20230403222550126](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230403222550126.png)



**调整$epoch=25$:**

训练集上$loss/train\_word$:

![image-20230407102329018](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230407102329018.png)

验证集上$loss/dev\_word$:

![image-20230407102532351](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230407102532351.png)

可以看出，即使训练集上的$loss$一直在减少，但是验证集中，当$epoch>=10$时，$loss$不降反升，最后趋于稳定，有可能**训练过度，导致过拟合**。

#### 困惑度

使用公式：
$$
ppl\left(\varepsilon_{\text {test }}\right)=e^{-W L L\left(\varepsilon_{\text {test }}\right)}
$$
即使用$read\_dataset()$读取数据后，再使用$calc\_sent\_loss(sent)$计算交叉熵损失，最后根据数学公式直接计算即可：

即使用

```python
print("id=%d loss=%f ppl=%f"%(send_id,loss.data, math.exp(loss.data/len(sent))))
```

最终得到
$$
ppl\left(\text{Jane went to the store}\right)=280.660836
$$

$$
ppl\left(\text{store to Jane went the}\right)=5823.000297
$$

运行结果：

![image-20230412154611140](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230412154611140.png)

#### 优化网络结构

结构如下：

![image-20230412100428073](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230412100428073.png)

训练集上$loss/train\_word$:

![image-20230412154737448](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230412154737448.png)

验证集上$loss/dev\_word$:

![image-20230412154818621](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230412154818621.png)



### 遇到的问题与解决方式

1. 

## 参考资料

- https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- https://blog.csdn.net/blmoistawinde/article/details/104966127





