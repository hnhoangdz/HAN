import torch
import sys
import csv
# csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

def mul_matrix(input, weight, bias=False, is_context=False):
    feature_list = []

    for feature in input:

        feature = torch.mm(feature, weight) 
        feature = torch.nan_to_num(feature, neginf=0.0, posinf=1.0)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.nan_to_num(feature, neginf=0.0,posinf=1.0)
        feature = torch.tanh(feature).unsqueeze(0)
        feature = torch.nan_to_num(feature, neginf=0.0,posinf=1.0)
        feature_list.append(feature)
        
    result = torch.cat(feature_list, 0)
    if is_context==True:
        result = result.squeeze(-1)
        
    return result

def mul_element_wise(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature = torch.nan_to_num(feature, neginf=0.0,posinf=1.0)
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list, sent_length_list = [], []
    
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for _, line in enumerate(reader):
            text = ""
            for txt in line[1:]:
                text += txt.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))
            # if len(sent_list) == 10:
            #     print(sent_list)
            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))
        
        sorted_length_word = sorted(word_length_list)
        sorted_length_sent = sorted(sent_length_list)
        # print(sorted_length_sent)
        max_length_word = sorted_length_word[int(0.8*len(sorted_length_word))]
        max_length_sent = sorted_length_sent[int(0.8*len(sorted_length_sent))]
    
    return max_length_word, max_length_sent

if __name__ == '__main__':
    data_path = '../data/ag_news_csv/test.csv'
    a,b = get_max_lengths(data_path)
    print(a,b)        
        
        