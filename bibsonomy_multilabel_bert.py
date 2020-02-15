import numpy as np
import operator
import os
from sklearn.model_selection import train_test_split
from simpletransformers.classification import MultiLabelClassificationModel
from simpletransformers.classification import classification_model
from multilabel_bert_util import transform_multilabel_as_multihot_new, acc_prec_rec_f1_hamming_scores, convert_2_df, binary_CE

#settings: data path and use_cuda
max_seq_length = 300
path = r'bibsonomy_preprocessed_merged_final.txt' ; use_cuda = True; fp16 = False # eddie server

with open(path, encoding="utf-8") as f_content:
    content = f_content.readlines()

texts = [line.strip().split('__label__')[0] for line in content]
multilabels = [line.strip().split('__label__')[1] for line in content]

#create label list
dict_label = {}
for labels in multilabels:
    for label in labels.split(' '):
        if dict_label.get(label, None) != None:
            dict_label[label] = dict_label[label] + 1
        else:
            dict_label[label] = 1
        
#sorted_x = sorted(dict_label.items(), key=operator.itemgetter(1))
#sorted_x.reverse
label_list_sorted = sorted(dict_label, key=dict_label.get, reverse=True)
print('unique labels:',len(label_list_sorted))

print('most frequent 10 code')
for i in range(10):
    print(label_list_sorted[i],dict_label[label_list_sorted[i]])

Y = [transform_multilabel_as_multihot_new(x.split(' '),label_list_sorted) for x in multilabels]
df = convert_2_df(texts,Y)
    
#maybe write a loop here for 10-fold cv later    
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=False) #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
print(train_df,len(train_df))

train_fold_df, valid_fold_df = train_test_split(train_df, test_size=0.1, shuffle=False)

#10 fold cv over train_df

#model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=len(label_list_sorted), args={'train_batch_size':4, 'overwrite_output_dir': True, 'max_seq_length': 512, 'fp16': False})
model = MultiLabelClassificationModel('bert', 'bert-base-uncased', num_labels=len(label_list_sorted), args={'train_batch_size':16, 'eval_batch_size':16,'overwrite_output_dir': True, 'max_seq_length': max_seq_length, 'fp16': False, 'num_train_epochs': 1, 'n_gpu': 2})

#simply load the trained model
#model = MultiLabelClassificationModel('bert', '/exports/eddie/scratch/hdong3/outputs', num_labels=len(label_list_sorted), args={'train_batch_size':16, 'eval_batch_size':16,'overwrite_output_dir': True, 'max_seq_length': max_seq_length, 'fp16': False, 'num_train_epochs': 1, 'n_gpu': 2})

# Train the model
model.train_model(train_fold_df)

result, model_outputs, wrong_predictions = model.eval_model(valid_fold_df,acc_prec_rec_f1_hamming_scores=acc_prec_rec_f1_hamming_scores)

print('valid_result:',result)
print('valid_model_outputs:',model_outputs)

# to_predict = train_fold_df['text'].tolist()
# preds, outputs = model.predict(to_predict)
# print('preds',preds)
# print('outputs',outputs)

result, model_outputs, wrong_predictions = model.eval_model(test_df,acc_prec_rec_f1_hamming_scores=acc_prec_rec_f1_hamming_scores)

print('valid_result:',result)
print('valid_model_outputs:',model_outputs)
#print('sum:',np.sum(model_outputs,axis=0),np.sum(model_outputs,axis=1))
