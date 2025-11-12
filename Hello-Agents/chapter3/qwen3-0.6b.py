import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")

# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入文本
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# print("编码后的输入文本:")
# print(model_inputs)

# 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的 Token ID
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)

# 下面开始调节采样参数,观察对输出的影响
# temperature: 控制生成文本的随机性，值越大，随机性越大，值越小，确定性越大
# top_p: 控制生成文本的多样性，值越大，多样性越大，值越小，确定性越大
# top_k: 控制生成文本的多样性，值越大，多样性越大，值越小，确定性越大

# 设置采样参数
temperature = 0.5
top_p = 0.9
top_k = 50

# 生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_new_tokens=512
)

# 解码生成的 Token ID
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("经过调整后的输出:")
print(response)

# 接下来尝试让模型进行信息抽取，设计并对比以下不同的提示策略（如Zero-shot、Few-shot、Chain-of-Thought）对输出结果的效果差异

info_text = "我的电话号码是1234567890，你的呢？"
# Zero-shot 提示
prompt_zero_shot = f"请提取以下文本中的所有数字: '{info_text}'"

# Few-shot 提示（示例）
prompt_few_shot = (
    "示例1: 文本：'他有2只猫和3条狗'，数字有：2, 3。\n"
    "示例2: 文本：'明天是2023年6月1日'，数字有：2023, 6, 1。\n"
    f"现在请提取文本：'{info_text}' 中的所有数字："
)

# Chain-of-Thought 提示
prompt_cot = (
    f"请一步一步思考，先找出'{info_text}'中的所有数字，然后以逗号分隔输出。"
)

prompts = [
    ("Zero-shot", prompt_zero_shot),
    ("Few-shot", prompt_few_shot),
    ("Chain-of-Thought", prompt_cot)
]

for strategy, prompt in prompts:
    messages = [
        {"role": "user", "content": prompt},
    ]
    # 模型输入编码
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
    # 生成
    output_ids = model.generate(input_ids, max_new_tokens=128)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[{strategy}]\n{prompt}\n模型输出：{response.strip()}")
