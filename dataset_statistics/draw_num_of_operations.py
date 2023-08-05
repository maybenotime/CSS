import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from feature_extraction import *

def operation_statistic(data):      
    ope_count = []      
    for refs in data:
        for sample in refs:
            operation = sample['operation']
            count = len(operation)
            ope_count.append(count)
     
    avg_ope = sum(ope_count)/len(ope_count)               
    return avg_ope,ope_count

data = load_data(data_path)
avg_ope,ope_array = operation_statistic(data)
print(avg_ope)
title = 'Number of simplification operations'

plt.subplots_adjust(bottom=0.125)
plt.hist(ope_array,bins=20,range=(0,5),density=True,align='left',cumulative=False,color='lime',edgecolor='black')
plt.xlabel(title,fontsize='xx-large',fontweight='bold')
plt.ylabel("density",fontsize='xx-large',fontweight='bold')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.title(title,fontsize='xx-large',fontweight='bold')
plt.savefig("../picture/{}".format(title))
plt.show()