# -*- coding: utf-8 -*-
"""
@author: shaked
"""

import pandas as pd
dtaa = pd.read_csv('DefaultOfCreditCardClients.csv')
print(dtaa.head(1))


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer 
from numpy import log2 as log
import pprint
from termcolor import colored
import scipy.stats
import random 
from pandas import DataFrame


# =============================================================================
# import and data processing 
# =============================================================================
def iniate_data():
    Cedit_Data = pd.read_csv('DefaultOfCreditCardClients.csv')
    Cedit_Data = Cedit_Data.drop('ID',1)
    return Cedit_Data


# =============================================================================
#  discretization transform the raw data
# =============================================================================
def continuous_variables(Cedit_Data):
    kbins = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
    continuous_var = ["LIMIT_BAL","AGE","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    for x in continuous_var:
          Cedit_Data[x] =  kbins.fit_transform(Cedit_Data[[x]])


# =============================================================================
# Decission Tree
# =============================================================================
def find_entropy(df):                #finding entropy in current staate
    Class = df.keys()[-1]            
    entropy = 0
    values = df[Class].unique()      #return the values of the target- T\F
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)   
    return entropy
  
  
def find_entropy_attribute(df,attribute):   #for every att finding the entropy 
  eps = np.finfo(float).eps
  Class = df.keys()[-1]                     
  target_variables = df[Class].unique()     #This gives all 1 and 0
  variables = df[attribute].unique()        #This gives different val in that attribute 
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])    # B()
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    if len(df.keys()) == 1:     #only target left
        return 'finish att'
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtree_val(df, node,value):
  return df[df[node] == value].reset_index(drop=True)

def sub_tree_att(df, node):
    df_less_att = df.drop(node,1)
    return df_less_att

def Tree(df,tree=None): 
    
    Class = df.keys()[-1]   
    #Here we build our decision tree
    num_of_T = df[df[Class] == 1].shape[0]
    num_of_F = df[df[Class] == 0].shape[0]
    #Get attribute with maximum information gain
    node = find_winner(df)
    if node == 'finish att':
        print(colored(f"there is no more att : sum of T is {num_of_T} sum of F is: {num_of_F}",'red'))
        return (num_of_T+num_of_F,num_of_T,num_of_F)    ##(s,p,n) 
        
    attValue = np.unique(df[node])      #Get distinct value of that attribute 
    key = (num_of_T+num_of_F,num_of_T,num_of_F,node)
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[key] = {}
        print(f"the Tree was empty so i add {node} ")
    


    for value in attValue:
        subtree = get_subtree_val(df,node,value)
        clValue,counts = np.unique(subtree[Class],return_counts=True)              
        if len(clValue)== 1: #Checking purity of subtree
            print(colored("subtree pure",'green'))
            if clValue[0] ==1:
                tree[key][value] = (counts[0],counts[0],0)  #(s,p,n)   
            else:
                tree[key][value] = (counts[0],0,counts[0])
        else:    
            print(colored(f"adding  to {node} in val {value} subtree to my Tree",'blue')) 
            tree[key][value] = Tree(sub_tree_att(subtree, node)) #Calling the function recursively 
    return tree





# =============================================================================
# pruning
# =============================================================================

def pruneLeaves(obj,father):
    global tree
    keys_and_p = []
    isLeaf = True
    for key in obj:
        if isinstance(obj[key], dict):
            for v in obj[key]:
                if isinstance(obj[key][v], dict):
                    isLeaf = False
                    parent = key
                    keys_and_p.append(v)
    if not keys_and_p :
        c = 0
        for node in obj:
            for leaf in obj[node]:
                pt = obj[node][leaf][0]*node[1]/node[0]
                pf = obj[node][leaf][0]*node[2]/node[0]
                p1 = obj[node][leaf][1]
                p2 = obj[node][leaf][2]
                delt1 =((p1 - pt)**2)/pt
                delt2 = ((p2 - pf)**2)/pf
                c = c + delt1 + delt2
        df = node[0] - 1
        critic = scipy.stats.chi2.ppf(.05, df = df)
        if c > critic:       # reject ho - dont do anything
            rem = key[3]
            newval = list(key)
            newval.remove(rem)
            newval = tuple(newval)
            print(colored(f"i pruned {key}",'red'))
            #obj[father] = newval
            return newval
        return obj
    else:
        for k in keys_and_p:
            obj[parent][k] =  pruneLeaves(obj[parent][k],k)
        return obj

# =============================================================================
# finding Answers
# =============================================================================
def finding_ans(sample,tree):
    while isinstance(tree,dict):
        Att_vec = next(iter(tree))
        Att = next(iter(tree))[3]
        value_in_samp = sample[Att]
        values = [v for v in tree] 
        try:
            tree = tree[Att_vec][value_in_samp]
        except KeyError:
            if next(iter(tree))[1] > next(iter(tree))[2]:
                return 1
            if next(iter(tree))[1] < next(iter(tree))[2]:
                return 0
            else:
                return  random.randint(0, 1)
    if tree[1]> tree[2]:
        return 1
    if tree[1] < tree[2]:
        return 0
    else: 
      if Att_vec[1]> Att_vec[2]:
        return 1
      if Att_vec[1] < Att_vec[2]:
         return 0
      else:
          return  random.randint(0, 1)
        
def answer(test,tree):
    ans_vec = list()
    for sample in range(len(test)):
        x = test.iloc[sample]
        ans = finding_ans(x,tree)
        ans_vec.append(ans)
    return ans_vec

def compare_col(pred,target):
    count = 0
    compare = pd.concat([pred, target],axis=1) 
    for i in range (len(compare)):
        if compare["pred"][i] == compare["Y"][i]:
            count = count + 1
    return count/len(compare)


def build_tree(k):
    Cedit_Data = iniate_data()
    #continuous variables to categorical because chi square test
    continuous_variables(Cedit_Data)
    # split the data to train and test
    #Cedit_Data = Cedit_Data.iloc[:500]
    train_data,test_data = train_test_split(Cedit_Data, test_size = k, random_state=123)
    MyTree = Tree(train_data,tree=None)
    print(" The tree before pruning is: ")
    pprint.pprint(MyTree)
    print()
    Tree_After_prun = pruneLeaves(MyTree,None)
    ans = answer(test_data,Tree_After_prun)
    pred = DataFrame (ans,columns=['pred'])
    target = list(test_data["default payment next month"])
    target = DataFrame(target,columns=['Y'])
    accuracy = compare_col(pred,target)
    print(f" the error is : {1-accuracy}")
    #chi square
    #print DecisionTree
    return Tree_After_prun,1- accuracy


def Average(lst): 
    return sum(lst) / len(lst)
#Before activating the function
# make sure that the data has passed all the stages of data processing
def tree_err(df,k):
    accuracy_list = list()
    x = int(len(df)/k)
    cells = split_dataframe(df,x)
    for i in range(k):
        val = pd.DataFrame()
        train = pd.DataFrame()
        for j in range(len(cells)-1):
            if j == i :
                val = cells[i]
            else:
                t =  cells[i]
                train = pd.concat([train,t])
        train = train.reset_index()
        MyTree = Tree(train,tree=None)
        Tree_After_prun = pruneLeaves(MyTree,None)
        ans = answer(val,Tree_After_prun)
        pred = DataFrame (ans,columns=['pred'])
        target = list(val["default payment next month"])
        target = DataFrame(target,columns=['Y'])
        accuracy = compare_col(pred,target)
        accuracy_list.append(accuracy)
    
    return 1 - Average(accuracy_list)

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks
    

def will_default(Exam_sample):
    credit_data = iniate_data()
    df_length = len(credit_data)
    credit_data.loc[df_length] = Exam_sample
    continuous_variables(credit_data)
    Exam_sample_After_PD = credit_data.loc[df_length]
    credit_data = credit_data.drop(credit_data.index[df_length-1])
    MyTree = Tree(credit_data,tree=None)
    Tree_After_prun = pruneLeaves(MyTree,None)
    ans  = finding_ans(Exam_sample_After_PD,Tree_After_prun)
    print(f"the answer is {ans}")
    return ans


def tree_error(k):
    credit_data = iniate_data()
    continuous_variables(credit_data)
    #credit_data = credit_data.iloc[:500]
    #s = split_dataframe(credit_data,int(len(credit_data)/k))
    error = tree_err(credit_data,k)
    print(f" the error is : {error}")
    return error



if __name__ == '__main__':
    from time import localtime, strftime, time
    DATA_TIME_FMT = "%d/%m/%Y %H:%M:%S"
    print(strftime(DATA_TIME_FMT, localtime()))
    LINE_SEP = "-" * 70 + '\n' + "-" * 70
    tick = time()
    a,b = build_tree(0.1)
    print(LINE_SEP)
    print(LINE_SEP)

    a = tree_error(3)
    print('tree error done')
    print(LINE_SEP)
    print(LINE_SEP)

    record_1 = [1,20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2,
                3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]

    record_2 = [1, 120000, 2, 2, 2, 26, -1, 2, 0, 0, 0, 2, 2682, 1725, 2682,
                3272, 3455, 3261, 0, 1000, 1000, 1000, 0, 2000]

    record_3 = [1, 90000, 2, 2, 2, 34, 0, 0, 0, 0, 0, 0, 29239, 14027, 13559,
                14331, 14948, 15549, 1518, 1500, 1000, 1000, 1000, 5000]
    a = will_default(record_1)
    print('the prediction of the first row is: ' + str(a) + ' Should have been 1')
    a = will_default(record_2)
    print('the prediction of the first row is: ' + str(a) + ' Should have been 1')
    a = will_default(record_3)
    print('the prediction of the first row is: ' + str(a) + ' Should have been 0')
    print(LINE_SEP)
    print(LINE_SEP)
    tok = time()
    def run_time_for_pipe(tick, tok):
        minutes, seconds = divmod(tok - tick, 60)
        time_msg = "{:0>2}:{:0>2}".format(int(minutes), int(seconds))
        msg = f"Total time for flower classification pipeline was {time_msg} (minutes:seconds)"
        print(msg)
    run_time_for_pipe(tick=tick, tok=tok)