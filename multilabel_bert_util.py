import numpy as np
import pandas as pd

#this is too slow
def transform_multilabel_as_multihot(label_list_str,label_list_sorted):
    #create indexing list
    label_list_binary = []
    for label in label_list_str:
        label_list_binary.append(label_list_sorted.index(label))
        
    result=np.zeros(len(label_list_sorted))
    #set those location as 1, all else place as 0.
    #label_list = [int(i) for i in label_list_binary]
    result[label_list_binary] = 1
    return result

def transform_multilabel_as_multihot_new(label_list_str,label_list_sorted):
    result = []
    for label in label_list_sorted:
        if label in label_list_str:
            result.append(1)
        else:
            result.append(0)
    return result

# convert to standard input for simpletransformers: DataFrame of 'text' (strings) and 'labels' (tuples of multihot representations).
def convert_2_df(trainX,trainY):
    trainY = [tuple(l) for l in trainY]
    list_of_tuples = list(zip(trainX, trainY))
    df = pd.DataFrame(list_of_tuples, columns = ['text', 'labels'])
    #df['labels'] = df['labels'].apply(lambda x: tuple(x))
    return df

#metrics, input labels as 2D list multihot representation and preds as 2D probabilities. 

def binary_CE(labels,preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    print('labels:',labels)
    print('preds:',preds)
    print('len(preds):',len(preds))
    return -(np.sum(np.multiply(np.log(preds), labels) + np.multiply((1 - labels), np.log(1 - preds))))/len(preds)
    
def acc_prec_rec_f1_hamming_scores(labels,preds):
    assert len(preds) == len(labels)
    assert len(preds[0]) == len(labels[0])
    acc, prec, rec, hamming_loss = 0.0, 0.0, 0.0, 0.0
    for i in range(len(preds)):
        labels_predicted = np.where(preds[i]>0.5)[0]
        #print(labels_predicted)
        curr_acc = calculate_accuracy(labels_predicted,labels[i])
        acc = acc + curr_acc
        curr_prec, curr_rec = calculate_precision_recall(labels_predicted,labels[i])
        prec = prec + curr_prec
        rec = rec + curr_rec
        curr_hl = calculate_hamming_loss(labels_predicted,labels[i])
        hamming_loss = hamming_loss + curr_hl
    acc = acc/float(len(preds))
    prec = prec/float(len(preds))
    rec = rec/float(len(preds))      
    hamming_loss = hamming_loss/float(len(preds))
    if prec+rec != 0:
        f_measure = 2*prec*rec/(prec+rec)
    else:
        f_measure = 0
    return acc,prec,rec,f_measure,hamming_loss/11.59
    
def calculate_accuracy(labels_predicted,labels):
    # turn the multihot representation to a list of true labels
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
        #print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    overlapping = 0
    label_dict = {x: x for x in label_nozero} # create a dictionary of labels for the true labels
    union = len(label_dict)
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            overlapping = overlapping + 1
        else:
            union = union + 1
    if union == 0:
        acc = 1
    else:
        acc = overlapping / union
    return acc

def calculate_precision_recall(labels_predicted, labels):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
    #    print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    if (len(labels_predicted)==0): # if nothing predicted, then set the precision as 0.
        precision=0
    else: 
        precision = count / len(labels_predicted)
    if len(label_nozero) != 0:
        recall = count / len(label_nozero)
    else:
        recall = 1
    #fmeasure = 2*precision*recall/(precision+recall)
    #print(count, len(label_nozero))
    return precision, recall
   
# calculate the symmetric_difference
def calculate_hamming_loss(labels_predicted, labels):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    count = 0
    label_dict = {x: x for x in label_nozero} # get the true labels

    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1 # get the number of overlapping labels
    
    return len(label_dict)+len(labels_predicted)-2*count
#print(transform_multilabel_as_multihot([2,4]))