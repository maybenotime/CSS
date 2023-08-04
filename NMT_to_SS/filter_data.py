from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

src_path = './news_commentary/news-commentary-v15.en-zh.tsv'           #中文，complex
tgt_path = './news_cn/news-en-ref-to-cn'           #翻译后的中文,simple
save_src = './train_data/complex_src'
save_tgt = './train_data/simple_tgt'

def load_src(path):
    src_data = []
    with open(path,"r") as f:
        for line in f:
            line = line.strip()
            lan_group = line.split('\t')
            if len(lan_group) != 2:
                src_data.append(lan_group[0])
            else:
                src_data.append(lan_group[1])
    return src_data

def load_tgt(path):
    tgt_data = []
    with open(path,"r") as f:
        for line in f:
            line = line.strip()
            tgt_data.append(line)
    return tgt_data

def filter(src,tgt,bleu_threshold=15):
    flag = True         #是否要过滤掉这条数据
    smooth = SmoothingFunction()
    BLEU_score = sentence_bleu([src],tgt,smoothing_function=smooth.method1) * 100
    if bleu_threshold < BLEU_score < 100:
        flag = False
    return flag

src_data = load_src(src_path)
tgt_data = load_tgt(tgt_path)

count = 0       #过滤后数据量
with open(save_src,"w") as w_src, open(save_tgt,"w") as w_tgt:
    for src,tgt in tqdm(zip(src_data,tgt_data)):
        flag = filter(src,tgt)
        if not flag:            #如果不用过滤数据
            count += 1
            print(src,file=w_src)
            print(tgt,file=w_tgt)

print(count)            #共计255407条数据