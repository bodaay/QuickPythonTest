
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Storage/DAMO-NLP-MT_polylm-13b", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/Storage/DAMO-NLP-MT_polylm-13b", device_map="auto")
model.eval()

input_doc = f"Beijing is the capital of China.\nTranslate this sentence from English to Arabic.\n\n"

inputs = tokenizer(input_doc, return_tensors="pt")
inputs = inputs.to('cuda')
generate_ids = model.generate(
  inputs.input_ids,
  attention_mask=inputs.attention_mask,
  do_sample=False,
  num_beams=4,
  max_length=128,
  early_stopping=True
)

decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

with open('output.txt', 'a', encoding='utf-8') as f:
    f.write(decoded)
### results
### Beijing is the capital of China.\nTranslate this sentence from English to Chinese.\\n北京是中华人民共和国的首都。\n ...