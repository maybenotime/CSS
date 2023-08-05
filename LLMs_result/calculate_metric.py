import json 
import jieba
from easse.sari import corpus_sari
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

dataset_path = '../CSS/test.json'
LLMs_result_path = './result/vicuna-fewshot.txt'
smooth = SmoothingFunction()

def tokenize(sen,char_level=False):         #默认是word_level，转换为用空格分词的字符串
    if char_level:
        char_list = list(sen)
        char_level_str = ' '.join(char_list)
        return char_level_str
    else:
        seg_list = jieba.lcut(sen)
        word_level_str  = ' '.join(seg_list)
        return word_level_str
    
with open(dataset_path,"r") as f:
    data = json.load(f)

with open(LLMs_result_path,"r") as f:
    result = []
    for line in f:
        line = line.strip()
        result.append(line)

def calculate(data,result):
    bleu_array = []
    sari_array = []
    for references,simplified_sen in zip(data,result):
        original = references[0]['source']
        print(original)
        print(simplified_sen)
        orignal_ = tokenize(original,char_level=False)
        sen_ = tokenize(simplified_sen,char_level=False)
        ref1 = tokenize(references[0]['target'][0],char_level=False)
        ref2 = tokenize(references[1]['target'][0],char_level=False)
        ref_list = [references[0]['target'][0],references[1]['target'][0]]
        bleu = sentence_bleu(ref_list,simplified_sen,smoothing_function=smooth.method1) * 100
        SARI_score = corpus_sari([orignal_],[sen_],[[ref1],[ref2]],use_f1_for_deletion=False,use_paper_version=True)
        bleu_array.append(bleu)
        sari_array.append(SARI_score)
    
    return bleu_array,sari_array



bleu_array,sari_array = calculate(data,result)
    
print("bleu:",sum(bleu_array)/len(bleu_array))
print("sari:",sum(sari_array)/len(sari_array))