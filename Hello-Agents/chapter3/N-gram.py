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
