from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import json
from tqdm import tqdm
import torch

test_path = './CSS/test.json'
result_path = './result/vicuna-zeroshot.txt'

LOAD_8BIT = True
tokenizer = LlamaTokenizer.from_pretrained("/data2/pretrain/vicuna/vicuna-13b")
model = LlamaForCausalLM.from_pretrained(
        "/data2/pretrain/vicuna/vicuna-13b",
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map='auto'
    )

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model = model.eval()


generation_config = dict(
    temperature=0.01,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=10,
    repetition_penalty=1.3,
    max_new_tokens=1024
    )

fewshot_template = '请在保留原意的基础上简化句子,以下是五个句子简化的示例\n\
原句：据了解，男子从嘉义北上，藉拾荒和行乞为生，除现身东区，也会到万华或中和寻求民众爱心。\n\
简化句：根据了解，男子从嘉义向北出发，靠捡东西和向人乞讨维持生活，除了出现在东区，也会到万华或中和来寻求民众的爱心。\n\
原句：盛夏吃的食物温度要稍低一些，避免吃滚烫的食物；寒冬吃的食物温度要稍高一些，不要吃冷食。\n\
简化句：夏天吃的食物温度要稍低一些，而寒冬吃的食物温度要稍高一些。\n\
原句：但现在，在投资者不断地质疑声中，同业竞争依然悬而未决——原本，这应该是上市前解决的问题。\n\
简化句：但现在，在投资者的质疑声中，同业竞争依然还没有解决——原本，这应该是上市前解决的问题。\n\
原句：在五脏与五行的关系中，黑色对应的是肾脏，而黑米性平、味甘，具有滋阴补肾、益气活血、暖肝明目的功效。\n\
简化句：在五脏与五行的关系中，黑色对应的是肾脏，而黑米药性平和、味道甜，具有滋阴补肾、益气活血、暖肝明目的功效。\n\
原句：这种黏糊的汁液就是白及的精华，可以涂抹在斑点、皱纹、鱼尾纹、眼袋的地方，能够起祛皱美容的作用。\n\
简化句：白及的精华有祛皱美容的作用，可以涂抹在斑点、皱纹、鱼尾纹、眼袋的地方。\n\
原句：{}\n\
简化句：'

def generate(input_text):
    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
    generation_output = model.generate(
        input_ids = inputs["input_ids"].to('cuda'), 
        attention_mask = inputs['attention_mask'].to('cuda'),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **generation_config
    )
    s = generation_output[0]
    output = tokenizer.decode(s,skip_special_tokens=True)
    response = output.split("简化句：")[-1].strip()
    return response

with open(test_path,"r") as f:
    data = json.load(f)

for entry in tqdm(data):
    sentence = entry[0]['source']
    instructs = "请在保留原意的基础上简化以下句子\n原句：{}\n简化句："
    prompt = instructs.format(sentence)
    print(instructs)
    response = generate(prompt)
    with open(result_path,"a",encoding='utf8') as w:
        if '\n' in response:            #过滤模型给出简化解释的情况
            re_list = response.split('\n')
            response = re_list[0]
        print(response)
        print(response,file=w)
