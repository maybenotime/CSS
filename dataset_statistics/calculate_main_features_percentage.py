import matplotlib.pyplot as plt
from feature_extraction import *

data = load_data(data_path)

#sentence_splitting
# avg_value,split_array = testset_feature_in_sentence_level(data,sentence_split)
# split_percentage = sum(split_array)/len(split_array)
# print(split_percentage)

#compression
# avg_value,compress_array = testset_feature_in_sentence_level(data,compression_level)
# count = 0
# for compress_ratio in compress_array:
#     if compress_ratio < 0.75:
#         count += 1
        
# print(count/len(compress_array))

#word reordering
# avg_value,reorder_array = testset_feature_in_sentence_level(data,get_reordered_words_proportion)
# print(reorder_array)
# count = 0
# for reorder_ratio in reorder_array:
#     if reorder_ratio != 0:
#         count += 1
# print(count/len(reorder_array))        

#word_deletion_only
# avg_value,delete_array = testset_feature_in_sentence_level(data,only_deleted_words)
# print(delete_array)
# count = 0
# for flag in delete_array:
#     if flag == True:
#         count += 1
# print(count/len(delete_array))    

#replace_only
# avg_value,distance_array = testset_feature_in_sentence_level(data,replace_only_levenshtein_distance)
# print(distance_array)
# count = 0
# for distance in distance_array:
#     if distance != 0:
#         count += 1
# print(count/len(distance_array))