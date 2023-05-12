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

if op=="train":
  for epoch in range(30):
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
    writer.add_scalar("train loss/word", train_loss/train_words, epoch)

    #验证集
    # Evaluate on dev set
    # set the model to evaluation mode
    model.eval()
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
      my_loss = calc_sent_loss(sent)
      dev_loss += my_loss.data
      dev_words += len(sent)

    # Keep track of the development accuracy and reduce the learning rate if it got worse
    if last_dev < dev_loss:
      optimizer.param_groups[0]['lr']/=2
    last_dev = dev_loss

    # Keep track of the best development accuracy, and save the model only if it's the best one
    if best_dev > dev_loss:
      torch.save(model, "model_epoch25.pt")

      best_dev = dev_loss

    # Save the model
    print("epoch %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (epoch, dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words/(time.time()-start)))
    writer.add_scalar("dev loss/word", dev_loss/dev_words, epoch)
    # Generate a few sentences
    for _ in range(5):
      sent = generate_sent()
      print(" ".join([i2w[x] for x in sent]))

  writer.close()
```

#### 更改

~~~python
#学习率learning_rate不能直接修改
if last_dev < dev_loss:
	optimizer.param_groups[0]['lr']/=2
~~~

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
def read_dataset(filename, add_vocab):
  with open(filename, "r") as f:
    for line in f:
      yield [get_wid(w2i, x, add_vocab) for x in line.strip().split(" ")]
def convert_to_variable(words):
  var = Variable(torch.LongTensor(words))
  if USE_CUDA:
    var = var.cuda()

  return var

# A function to calculate scores for one value
def calc_score_of_histories(words):
  # This will change from a list of histories, to a pytorch Variable whose data type is LongTensor
  words_var = convert_to_variable(words)
  logits = model(words_var)
  return logits

# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
  # The initial history is equal to end of sentence symbols
  hist = [S] * N
  # Step through the sentence, including the end of sentence token
  all_histories = []
  all_targets = []
  for next_word in sent + [S]:
    all_histories.append(list(hist))
    all_targets.append(next_word)
    hist = hist[1:] + [next_word]

  logits = calc_score_of_histories(all_histories)
  loss = nn.functional.cross_entropy(logits, convert_to_variable(all_targets), size_average=False)

  return loss
#Jane went to the store
#store to Jane went the
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

源代码$epoch=5$

**调整$epoch=25$:**

训练集上$loss/train\_word$:

![image-20230407102329018](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230407102329018.png)

验证集上$loss/dev\_word$:

![image-20230407102532351](C:\Users\LHA\AppData\Roaming\Typora\typora-user-images\image-20230407102532351.png)

可以看出，即使训练集上的$loss$一直在减少，但是验证集中，当$epoch>=10$时，$loss$不降反升，最后趋于稳定，有可能**训练过度，导致过拟合**。

#### 困惑度

| FNN_LM             | $ppl_1$ | $ppl_2$ |
| ------------------ | ------- | ------- |
| 未优化（epoch=5）  | 392     | 3092    |
| 未优化（epoch=25） | 313     | 2296    |
| 优化（epoch=25）   | 281     | 5823    |

可以看出，优化后的网络模型，"Jane went to the store"和“store to Jane went the”的困惑度差距最大，效果最后。下面困惑度计算以优化后为例。

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

- 优化后的网络结构略优于优化前的loss

- 同时测得的困惑度差距更大

  即使用更深层的网络，使用droupout()函数随机消失，可以使得FNN_LM结果更好。

### 遇到的问题与解决方式

1. 困惑度的计算，根据前面代码中ppl提示和数学公式可以算出

~~~python
print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (epoch, train_loss/train_words, math.exp(train_loss/train_words), train_words/(time.time()-start)))
~~~

​	2.网络结构改变，使用线性模型，网络层数过多会使得梯度消失，使用droupout函数

## 参考资料

- https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- https://blog.csdn.net/blmoistawinde/article/details/104966127





