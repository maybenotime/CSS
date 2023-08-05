import json 
from easse.sari import corpus_sari
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from feature_extraction import tokenize

dataset_path = '../dataset/test.json'
smooth = SmoothingFunction()

with open(dataset_path,"r") as f:
    data = json.load(f)
    

bleu_array = []
sari_array = []
for references in data:
    source = references[0]['source']
    ref1 = references[0]['target'][0]
    ref2 = references[1]['target'][0]
    source_ = tokenize(source)
    ref1_ = tokenize(ref1)
    ref2_ = tokenize(ref2)
    ref_list = [ref1_,ref2_]
    # ref1_bleu = sentence_bleu(ref2['target'],ref1['target'][0],smoothing_function=smooth.method1) * 100
    # ref2_bleu = sentence_bleu(ref1['target'],ref2['target'][0],smoothing_function=smooth.method1) * 100
    # bleu = (ref1_bleu + ref2_bleu)/2
    ref1_sari = corpus_sari([source_],[ref1_],[[ref2_]],use_f1_for_deletion=False,use_paper_version=True)
    ref2_sari = corpus_sari([source_],[ref2_],[[ref1_]],use_f1_for_deletion=False,use_paper_version=True)
    sari = (ref1_sari + ref2_sari)/2
    # bleu_array.append(bleu)
    sari_array.append(sari)

    
# print("gold_reference_bleu:",sum(bleu_array)/len(bleu_array))
print("gold_reference_sari:",sum(sari_array)/len(sari_array))