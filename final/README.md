# 语音识别

> 2013551 雷贺奥

## 语言模型部分

### 语言模型结果

实现了原始的模型即使用Gateconv进行编码后，使用GreedyDecoder进行解码后，作者对模型进行了训练，在epoch=100时，模型的CER最终收敛到了0.17。为了对其更进一步的优化，考虑使用Beam search decoder 和 使用了语言模型（LM）的Beam search decoder对模型进行优化。作者做了三组对照实验，最终结果如下：

|             模型              |         CER         |
| :---------------------------: | :-----------------: |
|       原模型（Greedy）        | 0.17008138439862533 |
|  原模型+beam search decoder   | 0.16985238835046104 |
| 原模型+LM+beam search decoder | 0.0987078061907305  |

由图可知，使用beam search decoder的结果要优于Greedy decoder，设词表大小为$N$，句子长度为$T$个单词

* Greedy算法：使用简单的贪心算法，即每一个状态节点都使用似然概率最大的character进行输出，时间复杂度为$O_{NT}$

  ,虽然时间复杂度较小，但很容易陷入局部最优解，使得模型的识别效果变差。

* Beam search算法：设置一个$K$,每一个状态节点都记录$K$条似然概率最大的路径，随后在与下个节点相连的$KN$条路径中选择$K$条概率最大的路径，直到递归到最后一个状态节点，每次都保留$K$条概率路径。时间复杂度为$O_{KNT}$可以看出，这种算法是为了减少时间复杂度，而求出的次优解，相较于暴力的全局最优解节省时间，相较于贪心算法不容易陷入局部最优。

最终的实验结果也证实了这一点，虽然beam search的优化效果较为一般，但还是可以看出其算法的优越性。

对beam search decoder加入语言模型后，模型的识别效果有了实质性的飞跃，CER大大降低，这是因为语言模型可以对文本进行纠错，同时语言模型在beam search decoder中具有自己的权重（$\alpha$）将在beam search decoder源码部分进行讲解。

总而言之，加入语言模型后，beam search decoder可以分辨例如同音字、错别字等情况（注：此时不是使用correct，而是将lm加入decoder）如：

```c
['炼刚产量大大提升']->['炼钢产量大大提升']
```

纠正同音字‘刚’和‘钢’，以提升CER，从而提升模型语音识别的效果。

### Beam Search 源码

下载Beam Search 库后，CTCBeamDecoder定义在`__init__.py`中，可是底层代码是使用C++写的，因此进入`ctc_beam_search_decoder.cpp`，进行阅读。

ctc_beam_search_decoder函数如下

~~~c++
std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer)
{
  DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id,
                     log_input, ext_scorer);
  state.next(probs_seq);
  return state.decode();
}
~~~

参数列表中`Scorer *ext_scorer`即使用LM辅助beam search decoder对输入的sequence进行打分，Scorer 的构造函数如下所示：

~~~c++
Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::vector<std::string>& vocab_list) {
  this->alpha = alpha;
  this->beta = beta;
  dictionary = nullptr;
  is_character_based_ = true;
  language_model_ = nullptr;
  max_order_ = 0;
  dict_size_ = 0;
  SPACE_ID_ = -1;
  setup(lm_path, vocab_list);
}
~~~

Scorer的参数列表中，alpha为语言模型打分的权重，beta为前缀序列的权重，前缀序列每加一，打分就加beta，lm_path就为存储语言模型的路径，vocab_list为decode时使用的字典。

ctc_beam_search_decoder函数调用的两个函数，含义如下：

（1）state.next(probs_seq)：使用Beam search正向计算最大似然概率的$K$条路径，同时加上beta，再加上alpha乘上语言模型所打的分数。更新参数需要分为以下三种情况

* 当前character预测为blank：更新log_prob_b_cur的值，即更新此条blank路径的似然概率。
* 当前character预测为前缀序列的最后一个: 由于character相同，需要合并，更新log_prob_nb_cur，即更新此条非blank路径的似然概率。
* 当前charater预测为新的单词：调用get_path_trie函数进行更新。

最终以上三种情况都会得出log_p，即Beam search 对sequence的打分，**接下来就是引入语言模型**。

~~~c++
DecoderState::next(const std::vector<std::vector<double>> &probs_seq)
{
    //省略。。。。。 只需要知道log_p 为 beam search 的打分即可
 			// language model scoring
        	if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }
            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            //alpha 语言模型权重
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            //beat 长度权重
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
	//省略。。。。。
}
~~~

至此，可以明了，CTC beam search decoder中是如何加载和使用语言模型的。

（2）state.decode()：阅读代码可知，通过next()所得出来的路径，反过来递归解析，从而得到需要的文本部分，至此，decorder的工作完成。



