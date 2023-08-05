import torch
import json
import jieba
import argparse
from easse.sari import corpus_sari
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data.dataset import WikiLargeZH,TEST_SET
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

use_predict_model = '/home/yangshiping/yangsp/projects/wikilarge_zh/mt5-baseline/summary_fewshot_finetune/checkpoint-116'      #预测时使用的模型的路径
save_output_path = './word_level/NMT_char_level'

save_model_dir = './model'
dev_path = '../our_data/dataset2/dev.json'          
test_path = '../our_data/dataset/test.json'
multitest_path = '../our_data/dataset/multi_references_test.json'

tokenizer = MT5Tokenizer.from_pretrained(use_predict_model)

def tokenize(sen,char_level=False):         #默认是word_level，转换为用空格分词的字符串
    if char_level:
        char_list = list(sen)
        char_level_str = ' '.join(char_list)
        return char_level_str
    else:
        seg_list = jieba.lcut(sen)
        word_level_str  = ' '.join(seg_list)
        return word_level_str

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", type=str, default=save_model_dir)   #保存模型的路径
    parser.add_argument("--batch_size_on_train", type=int, default=16)
    parser.add_argument("--batch_size_on_eval", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)    
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--wandb_running_name", type=str, default='test_mt5')
    args = parser.parse_args()
    return args



def load_dev_data(path):    #将json转成src和tgt的形式
    src_data = []
    tgt_data = []
    with open(path,"r") as f:
        data = json.load(f)
    for ins in data:
        src_data.append(ins['source'])
        tgt_data.append(ins['target'][0])
    
    return src_data,tgt_data

def load_test_data(path):      
    src_data = []
    references_data = []
    with open(path,"r") as f:
        data = json.load(f)
    for ins1,ins2 in data:
        src_data.append(ins1['source'])
        references = [ins1['target'][0],ins2['target'][0]]
        references_data.append(references)
    return src_data,references_data

def load_multiref_test(path):
    src_data = []
    references_data = []
    with open(path,"r") as f:
        data = json.load(f)
    for refs in data:
        src_data.append(refs[0]['source'])
        references = []
        for ref in refs:
            references.append(ref['target'][0])
        references_data.append(references)
    
    return src_data,references_data

'''
以下两个compute_metric函数分别用来计算验证集和测试集指标，计算一个时注释另一个
'''
# def compute_metrics(eval_preds):        #计算验证集指标
#     generate_result,labels = eval_preds
#     dev_src,dev_tgt = load_dev_data(dev_path)
#     smooth = SmoothingFunction()
#     BLEU_list = []
#     SARI_list = []
#     for output,reference,src in zip(generate_result,dev_tgt,dev_src):
#         output_str = tokenizer.decode(output, skip_special_tokens=True)
#         SARI_score = corpus_sari([src],[output_str],[[reference]],use_f1_for_deletion=False,use_paper_version=True)
#         BLEU_score = sentence_bleu([reference],output_str,smoothing_function=smooth.method1)
#         BLEU_list.append(BLEU_score)
#         SARI_list.append(SARI_score)
    
#     bleu = sum(BLEU_list)/len(BLEU_list)*100
#     sari = sum(SARI_list)/len(SARI_list)
#     return {
#             'bleu':bleu,
#             'sari':sari,
#     }

def compute_metrics(eval_preds):        #计算测试集指标
    generate_result,labels = eval_preds
    test_src,test_ref = load_test_data(test_path)       #测试集
    # test_src,test_ref = load_multiref_test(multitest_path)      #多参考测试集
    smooth = SmoothingFunction()
    BLEU_list = []
    SARI_list = []
    for output,references,src in zip(generate_result,test_ref,test_src):
        output_str = tokenizer.decode(output, skip_special_tokens=True)
        SARI_score = corpus_sari([tokenize(src)],[tokenize(output_str)],[[tokenize(ref)] for ref in references],use_f1_for_deletion=False,use_paper_version=True)
        BLEU_score = sentence_bleu(references,output_str,smoothing_function=smooth.method1)
        BLEU_list.append(BLEU_score)
        SARI_list.append(SARI_score)
    
    bleu = sum(BLEU_list)/len(BLEU_list)*100
    sari = sum(SARI_list)/len(SARI_list)
    return {
            'bleu':bleu,
            'sari':sari,
    }



def main(args):
    model = MT5ForConditionalGeneration.from_pretrained(use_predict_model)
    dev_src,dev_tgt = load_dev_data(dev_path)
    dev_dataset = WikiLargeZH(dev_src,dev_tgt,tokenizer)
    test_src,test_ref = load_test_data(test_path)
    test_dataset = TEST_SET(test_src,test_ref,tokenizer)
    multi_src,multi_ref = load_multiref_test(multitest_path)
    multiref_test_dataset = TEST_SET(multi_src,multi_ref,tokenizer)
    
    args = Seq2SeqTrainingArguments(                        
        output_dir=args.output_model_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_on_train,
        per_device_eval_batch_size=args.batch_size_on_eval,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=2,                #checkpoints中最多会保留几个模型  
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,        #计算生成内容的指标时需要置为True 
        save_strategy='epoch', 
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
        metric_for_best_model='sari',         #自定义指标选择模型
        # include_inputs_for_metrics=True,      #当前版本无法使用，直接load_dev_data来解决
        run_name=args.wandb_running_name,
        logging_dir='./logs',
        generation_max_length=128,
        generation_num_beams=10,
    )

    
    data_collator = DataCollatorForSeq2Seq(tokenizer,model,padding=True)  #类似于pytorch中的dataloader

    trainer = Seq2SeqTrainer(
        model,
        args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics
    )

    torch.backends.cudnn.enabled = False
    predictions,label,metric = trainer.predict(test_dataset=multiref_test_dataset,num_beams=10)         #进行预测
    result = tokenizer.batch_decode(predictions,skip_special_tokens=True)
    
    print(metric)
    with open(save_output_path,"w") as w:           #存储输出
        print(metric,file=w)
        for output in result:
            print(output,file=w)



if __name__ == '__main__':
    arg = args()
    main(arg)