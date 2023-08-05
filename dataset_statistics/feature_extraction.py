import json
import jieba
import Levenshtein
from similarity.normalized_levenshtein import NormalizedLevenshtein     #换这个包算
import numpy as np
from collections import Counter
from functools import lru_cache
from zhon.hanzi import punctuation
from pyltp import SentenceSplitter

data_path = '../dataset/test.json'
word_frequency_file = '../count.out'
stop_words_path = '../stopwords.txt'


def tokenize(sen,char_level=False):         #默认是word_level，转换为用空格分词的字符串
    if char_level:
        char_list = list(sen)
        char_level_str = ' '.join(char_list)
        return char_level_str
    else:
        seg_list = jieba.lcut(sen)
        word_level_str  = ' '.join(seg_list)
        return word_level_str

def load_data(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

def to_words(sen):            #转化为分词后的列表
    return tokenize(sen).split(' ')

def count_words(word_list):
    return len(word_list)
    
def get_levenshtein_distance(complex_sentence, simple_sentence):
    normalized_levenshtein = NormalizedLevenshtein()
    distance = normalized_levenshtein.distance(complex_sentence,simple_sentence)
    return distance

def flatten_counter(counter):
    return [k for key, count in counter.items() for k in [key] * count]



def get_added_words(c, s):
    return flatten_counter(Counter(to_words(s)) - Counter(to_words(c)))


def get_deleted_words(c, s):
    return flatten_counter(Counter(to_words(c)) - Counter(to_words(s)))


def get_kept_words(c, s):
    return flatten_counter(Counter(to_words(c)) & Counter(to_words(s)))


def get_lcs(seq1, seq2):
    '''Returns the longest common subsequence using memoization (only in local scope)'''
    @lru_cache(maxsize=None)
    def recursive_lcs(seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return []
        if seq1[-1] == seq2[-1]:
            return recursive_lcs(seq1[:-1], seq2[:-1]) + [seq1[-1]]
        else:
            return max(recursive_lcs(seq1[:-1], seq2), recursive_lcs(seq1, seq2[:-1]), key=lambda seq: len(seq))

    try:
        return recursive_lcs(tuple(seq1), tuple(seq2))
    except RecursionError as e:
        print(e)
        # TODO: Handle this case
        return []


def get_reordered_words(c, s):
    # A reordered word is a word that is contained in the source and simplification
    # but not in the longuest common subsequence
    lcs = get_lcs(to_words(c), to_words(s))
    return flatten_counter(Counter(get_kept_words(c, s)) - Counter(lcs))


def get_n_added_words(c, s):
    return len(get_added_words(c, s))


def get_n_deleted_words(c, s):
    return len(get_deleted_words(c, s))


def get_n_kept_words(c, s):
    return len(get_kept_words(c, s))


def get_n_reordered_words(c, s):
    return len(get_reordered_words(c, s))


def get_added_words_proportion(c, s):
    # TODO: Duplicate of get_addition_proportion, same for deletion
    # Relative to simple sentence
    return get_n_added_words(c, s) / count_words(s)


def get_deleted_words_proportion(c, s):
    # Relative to complex sentence
    return get_n_deleted_words(c, s) / count_words(c)


def get_reordered_words_proportion(c, s):
    # Relative to complex sentence
    return get_n_reordered_words(c, s) / count_words(c)


def only_deleted_words(c, s):
    # Only counting deleted words does not work because sometimes there is reordering
    return  get_lcs(to_words(c), to_words(s)) == to_words(s)

@lru_cache(maxsize=None)
def get_word2rank(path=word_frequency_file,vocab_size=30000):
    word2rank = {}
    with open(path,"r",encoding='gbk') as f:
        for rank,line in enumerate(f):
            if rank > vocab_size:
                break
            line = line.strip()
            word,_ = line.split(' ')
            word2rank[word] = rank
    return word2rank

def is_punctuation(word):
    return word in punctuation

def remove_punctuation_tokens(text):
    return ''.join([w for w in to_words(text) if not is_punctuation(w)])

@lru_cache(maxsize=None)
def get_stopwords(path=stop_words_path):
    stop_words = []
    with open(path,"r")  as f:
        for line in f:
            line = line.strip()
            stop_words.append(line)
    return stop_words

def remove_stopwords(text):
    return ''.join([w for w in to_words(text) if w not in get_stopwords()])

def get_rank(word):
    return get_word2rank().get(word, len(get_word2rank()))

def get_log_rank(word):
    return np.log(1 + get_rank(word))

def get_wordrank_score(sentence):               #todo:计算词汇复杂度，和论文中的描述一致
    # Computed as the third quartile of log ranks
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]
    score_array = []
    for word in words:
        squared_score = get_log_rank(word)**2
        score_array.append(squared_score)
        
    return sum(score_array)/len(score_array)

def get_lexical_complexity_score_ratio(org,tgt):
    org_score = get_wordrank_score(org)
    tgt_score = get_wordrank_score(tgt)
    
    return tgt_score/org_score

def compression_level(org,tgt):  #简化句的字符数除以原句的字符数
    count_org = len(to_words(org))     #以空格为分词符
    count_tgt = len(to_words(tgt))
    return count_tgt/count_org

def replace_only_levenshtein_distance(org,tgt):            #按照asset原文进行计算
    replace_ops = Levenshtein.distance(org,tgt,weights=(1,1,2)) - Levenshtein.distance(org,tgt,weights=(1,1,1))
    min_len = min(len(org),len(tgt))
    return replace_ops/min_len

def sentence_split(org,tgt):
    sents_org = SentenceSplitter.split(org)
    sents_tgt = SentenceSplitter.split(tgt)
    return len(sents_tgt) - len(sents_org)
    
def testset_feature_in_sentence_level(data,feature_function): #这个函数有问题
    feature_array = []
    for references in data:
        ref1 = references[0]    
        ref2 = references[1]
        org = ref1['source']
        tgt1 = ref1['target'][0]
        tgt2 = ref2['target'][0]
        feature_array.append(feature_function(org,tgt1))
        feature_array.append(feature_function(org,tgt2))

    return sum(feature_array)/len(feature_array),feature_array


