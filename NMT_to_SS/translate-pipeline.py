from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline
from torch.utils.data import Dataset
from tqdm import tqdm

file_path = './news_commentary/news-commentary-v15.en-zh.tsv'
save_path = './news_cn/news-en-ref-to-cn'

def load_data(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            line = line.strip()
            lan_group = line.split('\t')
            data.append(lan_group[0])
    return data


mode_name = 'liam168/trans-opus-mt-en-zh'
model = AutoModelWithLMHead.from_pretrained(mode_name,cache_dir='./translation_model')
tokenizer = AutoTokenizer.from_pretrained(mode_name,cache_dir='./translation_model')
translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer,device=0)

class mydataset(Dataset):
    def __init__(self,data):
        self.data = data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]

data = load_data(file_path)
dataset = mydataset(data)

with open(save_path,"w") as w:
    for output in tqdm(translation(dataset, max_length=256, batch_size = 32),total=len(dataset)):
        print(output[0]['translation_text'],file=w)

