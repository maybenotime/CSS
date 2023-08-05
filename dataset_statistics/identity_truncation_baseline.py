import json 
from easse.sari import corpus_sari
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from feature_extraction import tokenize

dataset_path = '../dataset/test.json'
smooth = SmoothingFunction()

with open(dataset_path,"r") as f:
    data = json.load(f)
    

def truncation_data(data):
    bleu_array = []
    sari_array = []
    for references in data:
        original = references[0]['source']
        print(original)
        Truncation = references[0]['source'][:int(len(original)*0.8)]
        print(Truncation)
        orignal_ = tokenize(original,char_level=False)
        Truncation_ = tokenize(Truncation,char_level=False)
        ref1_no = references[0]['target'][0]
        ref2_no = references[1]['target'][0]
        ref_list_no = [ref1_no,ref2_no]
        ref1 = tokenize(references[0]['target'][0],char_level=False)
        ref2 = tokenize(references[1]['target'][0],char_level=False)
        ref_list = [ref1,ref2]
        bleu = sentence_bleu(ref_list_no,Truncation,smoothing_function=smooth.method1) * 100
        SARI_score = corpus_sari([orignal_],[Truncation_],[[ref1],[ref2]],use_f1_for_deletion=False,use_paper_version=True)
        bleu_array.append(bleu)
        sari_array.append(SARI_score)
    
    return bleu_array,sari_array



bleu_array,sari_array = truncation_data(data)
    
print("Truncation_bleu:",sum(bleu_array)/len(bleu_array))
print("Truncation_sari:",sum(sari_array)/len(sari_array))