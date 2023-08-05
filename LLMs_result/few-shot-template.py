import json
import random
'''
随机抽取句子构成few-shot template
'''

additional_dataset = '../CSS/additional_dataset_for_few-shot_setting.json'

with open(additional_dataset,"r") as f:
    data = json.load(f)

sample_list = []
for i in range(5):
    sample = random.choice(data)
    sample_list.append(sample)

print(sample_list)

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

print(fewshot_template)