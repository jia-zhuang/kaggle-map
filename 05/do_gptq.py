import pandas as pd
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "./iter_0002752/"
data_path = '../../datasets/gen_cls/valid_split_message.jsonl'
quant_path = "./gen_cls_32b_full_qptq_4bit"



# prepare data
tokenizer = AutoTokenizer.from_pretrained(model_id)

calib_df = pd.read_json(data_path, lines=True)
calib_df = calib_df.sample(n=1024).copy()

calib_df['text'] = calib_df.messages.apply(
    tokenizer.apply_chat_template,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True    # 只在infer时有效
)

# remove think
calib_df['text'] = calib_df['text'].apply(lambda x: x.replace('<think>\n\n</think>\n\n', ''))

print(calib_df.sample().iloc[0].text)


# Quantize
quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calib_df.text.tolist(), batch_size=32)
model.save(quant_path)

# # test post-quant inference
# model = GPTQModel.load(quant_path)
# result = model.generate("Uncovering deep insights begins with")[0] # tokens
# print(model.tokenizer.decode(result)) # string output
