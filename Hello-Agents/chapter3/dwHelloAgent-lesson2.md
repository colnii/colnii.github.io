transformer架构还是手撕不了，上班边摸鱼边学还是学了太久了
# 习题
---
1) 
  ```python
  import collections

  # 示例语料库，与上方案例讲解中的语料库保持一致
  corpus = "datawhale agent learns datawhale agent works"
  tokens = corpus.split()
  total_tokens = len(tokens)
  
  # --- 第一步:计算 P(agent) ---
  count_agent = tokens.count('agent')
  p_agent = count_agent / total_tokens
  print(f"第一步: P(agent) = {count_agent}/{total_tokens} = {p_agent:.3f}")
  
  # --- 第二步:计算 P(works|agent) ---
  # 先计算 bigrams 用于后续步骤
  bigrams = list(zip(tokens, tokens[1:]))
  # print(bigrams)
  bigram_counts = collections.Counter(bigrams)
  count_agent_works = bigram_counts[('agent', 'works')]
  # count_agent 已在第一步计算
  p_agent_given_works = count_agent_works / count_agent
  print(f"第二步: P(works|agent) = {count_agent_works}/{count_agent} = {p_agent_given_works:.3f}")
  
  # --- 最后:将概率连乘 ---
  p_sentence = p_agent * p_agent_given_works
  print(f"最后: P('agent works') ≈ {p_agent:.3f} * {p_agent_given_works:.3f} = {p_sentence:.3f}")
  ```
  这个假设的含义是语言中每个词出现的概率取决于前面有限范围内所说过的词
  根本性局限是，这套架构本身并不理解词之间的含义，无法跳出所给出的语料库的内容
  RNN/LSTM提供了怎么保存记忆/如何保存记忆的网络架构思路，在记忆上进行了突破，
  Transformer则基于词向量的基础上确定了一个富有潜力的网络架构，在理解能力上进行了突破

2)
  核心思想：1.语言存在着因果关系；2.词汇是在运动中不断变化并展现其内涵的；3.句子中每个词语的重要性并不等同
  并行处理主要归功于mha，也有可能是我的理解不行，但RNN这个架构确实缺乏并行的能力。。位置编码就像是一个插件，让模型有机会依据这个差异学习到语言的先后顺序是有意义的
  区别在于少了交叉注意力和计算交叉注意力的encoder架构（感觉在说废话）。采用的原因，从数学上来讲，交叉注意力不加掩码，这会导致计算量剧增，再加上从任务角度讲交叉注意力不是必要的。

3)
    因为以字符和单词为输入单元会导致模型稀疏化，从而一方面需要更多的数据才能实现相同大小的架构性能
    bpe算法解决了怎么找到最常用的词汇片段，并将其科学的压缩到数字的问题

4)
    

5)
    RAG: 工作原理:调用向量数据库，把符合提问的知识塞到大模型的上下文
         使用场景：在一个专业场景具备大量各种多模态、结构化和半结构化的资料时，使用RAG能够最大化提升ai在这方面回答的准确性和能力
    其他缓解办法：把知识存储在KV cache里 优势：回答速度特别快

6)
    论文场景多以英语为主，且因为论文的前沿性，对模型能力有高要求->选择gpt、gemini
    切块之后一遍读一遍写readme.md，或者设置token_limit，到了就总结一下压缩上下文
    emmmm把ai生成的信息转化成查询再跑一遍RAG查询，查询完让模型再比对，如果因为相似度太低没过score阈值就说明在编造，需要重新思考
    
