import json

data_path = '../dataset/test.json'

def load_data(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

def operation_statistic(data):      
    ope_count = {}
    divide_by_ope = {}      
    total_ope = 0
    for refs in data:
        for sample in refs:
            operation = sample['operation']
            total_ope += len(operation)
            for ope in operation:
                if ope not in ope_count:
                    divide_by_ope[ope] = []
                    divide_by_ope[ope].append(sample)
                    ope_count[ope] = 1
                else:
                    ope_count[ope] += 1
                    divide_by_ope[ope].append(sample)
                    
    return total_ope,ope_count,divide_by_ope

data = load_data(data_path)
total_ope,ope_count,divde_by_ope = operation_statistic(data)

for key,value in divde_by_ope.items():
    dict = {}
    for sample in value:
        dict[sample['source']] = 1
    print(key,len(dict))
