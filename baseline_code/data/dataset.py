from torch.utils.data import Dataset
from tqdm import tqdm

class WikiLargeZH(Dataset):
    def __init__(self,src_data,tgt_data,tokenizer):
        super().__init__()
        self.model_inputs = []
        for src,tgt in tqdm(zip(src_data,tgt_data)):
            example = {}
            input_ids = tokenizer(src,max_length=128,padding=True,truncation=True)
            decoder_outputs = tokenizer(tgt,max_length=128,padding=True,truncation=True)    
            example['input_ids'] = input_ids['input_ids']      
            example['labels'] = decoder_outputs['input_ids']
            self.model_inputs.append(example)
            
    def __len__(self):
        return len(self.model_inputs)
    
    def __getitem__(self,index):
        return self.model_inputs[index]

class TEST_SET(Dataset):
    def __init__(self,src_data,ref_data,tokenizer):
        super().__init__()
        self.model_inputs = []
        for src,refs in tqdm(zip(src_data,ref_data)):
            example = {}
            input_ids = tokenizer(src,max_length=128,padding=True,truncation=True)
            decoder_outputs = tokenizer(refs[0],max_length=128,padding=True,truncation=True)    #随便取个label
            example['input_ids'] = input_ids['input_ids']      
            example['labels'] = decoder_outputs['input_ids']
            self.model_inputs.append(example)
            
    def __len__(self):
        return len(self.model_inputs)
    
    def __getitem__(self,index):
        return self.model_inputs[index]
        
            
