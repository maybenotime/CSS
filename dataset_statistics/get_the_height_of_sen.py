from ltp import LTP

ltp = LTP()  # 默认加载 Small 模型

def get_tree(dep):
    tree = {}
    for tup in dep[0]:
        child = tup[0]
        father = tup[1]
        if father not in tree:
            tree[father] = [child] 
        else:
            tree[father].append(child)
    return tree

def get_depth(root,tree):                #递归计算树深度
    if root not in tree:            #叶子节点
        return 0
    max = 0                         #子树的最大深度
    for child in tree[root]:        #遍历孩子节点
        m = get_depth(child,tree) 
        if m > max:
            max = m
    max += 1                    #子树遍历完，+1返回上一层
    return max 


def get_height_loop(sen):          #把前面的函数封装起来,sen是字符串
    seg, hidden = ltp.seg([sen])
    dep_info = ltp.dep(hidden)         #依存句法分析结果
    tree = get_tree(dep_info)          #建树
    height = get_depth(0,tree)
    
    return height

def get_depth_ratio(org,tgt):
    tgt_height = get_height_loop(tgt)
    org_height = get_height_loop(org)
    print(1)
    ratio = tgt_height/org_height
    return ratio
# height = get_height_loop('他叫汤姆去拿外衣')
# print(height)
    