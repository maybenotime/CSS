import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from feature_extraction import *
from get_the_height_of_sen import *

# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,sentence_split)
# # print(feature_array)
# title = 'Sentence Splits'
# print(avg_value)

# plt.hist(feature_array,bins=20,range=(-1,4),density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel("sentence splits",fontsize='x-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title(title,fontsize='xx-large',fontweight='bold')
# plt.savefig("../picture/{}".format(title))
# plt.show()

# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,compression_level)
# # print(feature_array)
# title = 'Compression levels'
# print(avg_value)

# plt.subplots_adjust(bottom=0.125)
# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel(title,fontsize='xx-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("../picture/{}".format(title))
# plt.show()

# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,get_levenshtein_distance)
# # print(feature_array)
# title = 'Levenshtein Distance'
# print(avg_value)

# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel("levenshtein distance",fontsize='x-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title(title,fontsize='xx-large',fontweight='bold')
# plt.savefig("../picture/{}".format(title))
# plt.show()

# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,replace_only_levenshtein_distance)
# # print(feature_array)
# title = 'Replace-only Levenshtein Distance'
# print(avg_value)


# plt.subplots_adjust(bottom=0.125)
# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel(title,fontsize='xx-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("../picture/{}".format(title))
# plt.show()


# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,get_deleted_words_proportion)
# # print(feature_array)
# title = 'Proportion of words deleted'
# print(avg_value)


# plt.subplots_adjust(bottom=0.125)
# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel(title,fontsize='xx-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("../picture/{}".format(title))
# plt.show()


# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,get_added_words_proportion)
# # print(feature_array)
# title = 'Proportion of words added'
# print(avg_value)


# plt.subplots_adjust(bottom=0.125)
# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel(title,fontsize='xx-large',fontweight='bold')
# plt.ylabel("density",fontsize='xx-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("../picture/{}".format(title))
# plt.show()


# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,get_reordered_words_proportion)
# # print(feature_array)
# title = 'Proportion of words reordered'
# print(avg_value)

# plt.subplots_adjust(bottom=0.125)
# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='lime',edgecolor='black')
# plt.xlabel(title,fontsize='xx-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("../picture/{}".format(title))
# plt.show()

data = load_data(data_path)
avg_value,feature_array = testset_feature_in_sentence_level(data,get_lexical_complexity_score_ratio)
# print(feature_array)
title = 'Lexical complexity score ratio'
print(avg_value)

plt.subplots_adjust(bottom=0.125)
plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='grey',edgecolor='black')
plt.xlabel(title,fontsize='xx-large',fontweight='bold')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("../picture/{}".format(title))
plt.show()



# data = load_data(data_path)
# avg_value,feature_array = testset_feature_in_sentence_level(data,get_depth_ratio)
# # print(feature_array)
# title = 'Dependency tree depth ratio'
# print(avg_value)

# plt.subplots_adjust(bottom=0.125)
# plt.hist(feature_array,bins=20,density=True,align='left',cumulative=False,color='grey',edgecolor='black')
# plt.xlabel(title,fontsize='xx-large',fontweight='bold')
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))        #纵坐标只显示整数
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("../picture/tree_depth_ratio")
# plt.show()