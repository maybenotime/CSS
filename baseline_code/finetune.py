import torch
import json
import wandb
import argparse
from easse.sari import corpus_sari
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data.dataset import WikiLargeZH
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

save_model_dir = './new_wikilargeZH_model'
src_path = '../our_data/zh/wiki.full.aner.ori.train.src'
tgt_path = '../our_data/zh/wiki.full.aner.ori.train.dst'
dev_path = '../our_data/dataset2/dev.json'          #验证集

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base',cache_dir='./mt5-base')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", type=str, default=save_model_dir)   #保存模型的路径
    parser.add_argument("--batch_size_on_train", type=int, default=32)
    parser.add_argument("--batch_size_on_eval", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)    
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--wandb_running_name", type=str, default='test_mt5')
    args = parser.parse_args()
    return args

def load_train_data(path):        #加载训练数据
    data = []
    with open(path,"r") as f:       
        for line in f:
            line = line.strip()
            data.append(line)
    
    return data

def load_dev_data(path):    #将json转成src和tgt的形式
    src_data = []
    tgt_data = []
    with open(path,"r") as f:
        data = json.load(f)
    for ins in data:
        src_data.append(ins['source'])
        tgt_data.append(ins['target'][0])
    
    return src_data,tgt_data

def compute_metrics(eval_preds):        #sari指标的计算有些问题
    generate_result,labels = eval_preds
    dev_src,dev_tgt = load_dev_data(dev_path)
    smooth = SmoothingFunction()
    BLEU_list = []
    SARI_list = []
    for output,reference,src in zip(generate_result,dev_tgt,dev_src):
        output_str = tokenizer.decode(output, skip_special_tokens=True)
        SARI_score = corpus_sari([src],[output_str],[[reference]],use_f1_for_deletion=False,use_paper_version=True)
        #注意此处的sari计算脚本经过修改，中英文分词方式不同。目前计算版本是严格按照论文来的,字级别ngram
        BLEU_score = sentence_bleu([reference],output_str,smoothing_function=smooth.method1)
        BLEU_list.append(BLEU_score)
        SARI_list.append(SARI_score)
    
    bleu = sum(BLEU_list)/len(BLEU_list)*100
    sari = sum(SARI_list)/len(SARI_list)
    return {
            'bleu':bleu,
            'sari':sari,
            'mix_score':bleu+sari
    }



def main(args):
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base',cache_dir='./mt5-base')
    train_src = load_train_data(src_path)
    train_tgt = load_train_data(tgt_path)
    dev_src,dev_tgt = load_dev_data(dev_path)
    train_dataset = WikiLargeZH(train_src,train_tgt,tokenizer)
    dev_dataset = WikiLargeZH(dev_src,dev_tgt,tokenizer)
    wandb.init(project="mt5_base_wikilarge")
    
    #配置trainer参数
    args = Seq2SeqTrainingArguments(                        
        output_dir=args.output_model_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=800,        
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_on_train,
        per_device_eval_batch_size=args.batch_size_on_eval,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,                #checkpoints中最多会保留几个模型  
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,        #计算生成内容的指标时需要置为True 
        save_strategy='steps',
        save_steps=800, 
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
        metric_for_best_model='sari',         #自定义指标选择模型
        # include_inputs_for_metrics=True,      #当前版本无法使用，直接load_dev_data来解决
        run_name=args.wandb_running_name,
        report_to='wandb',                      #报告结果和日志的平台
        logging_dir='./logs',
        generation_max_length=128,
        generation_num_beams=10,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer,model)  #类似于pytorch中的dataloader

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics
    )
    
    trainer.train()         #进行微调




if __name__ == '__main__':
    arg = args()
    main(arg)
    

