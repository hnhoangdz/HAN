import torch
import numpy as np
import pandas as pd
from src.HAN_net import HANnet
from src.utils import get_max_lengths
from nltk.tokenize import sent_tokenize, word_tokenize
import torch.nn.functional as F
import argparse
import csv

def get_args():
    parser = argparse.ArgumentParser(
        """Inference of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--data_path", type=str, default="data/ag_news_csv/test.csv")
    parser.add_argument("--raw_text", type=str, default="AI is a modern technology. It's a new electricity")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/best_model_10_0.22149476358870498.pth")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    args = vars(parser.parse_args())

    return args


def load_model(model_path, model, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

def doc_tokenizer(document, dictionary_path, max_length_sentence, max_length_word):
    dictionary = pd.read_csv(filepath_or_buffer=dictionary_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values
    dictionary = [word[0] for word in dictionary]
    
    document_encode = [[dictionary.index(word.lower()) if word.lower() in dictionary else -1 
                        for word in word_tokenize(sentences)]
                       for sentences in sent_tokenize(document)]

    # padding for sentence
    for sent_encode in document_encode:
        if len(sent_encode) < max_length_word:
            extend_word_encode = [-1]*(max_length_word - len(sent_encode))
            sent_encode.extend(extend_word_encode)
    
    # padding for document
    if len(document_encode) < max_length_sentence:
        extend_sent_encode = [[-1]*max_length_word]*(max_length_sentence - len(document_encode))
        document_encode.extend(extend_sent_encode)
    document_encode = [sentences[:max_length_word] for sentences in document_encode[:max_length_sentence]]
    
    document_encode = np.stack(document_encode, axis=0)
    document_encode += 1
    
    # empty_array = np.zeros_like(document_encode, dtype=np.int64)
    # input_array = np.stack([document_encode, empty_array], axis=0)
    
    feature = torch.from_numpy(document_encode)
    feature = feature.unsqueeze(0)
    return feature

def predict(feature, model, class_names):
    
    if torch.cuda.is_available():
        feature = feature.cuda()
        
    model.eval()
    with torch.no_grad():
        model._init_hidden_state(1)
        prediction = model(feature)
    print(prediction)
    prediction = F.softmax(prediction, dim=-1)
    print(prediction)
    prob, id = torch.max(prediction, dim=-1)
    label = class_names[id]
    
    return prob, label

def get_test_documents(test_path):
    documents = []
    labels = []
    
    with open(test_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for _, line in enumerate(reader):
            # merge title and description as a document
            text = ""
            for txt in line[1:]:
                text += txt.lower()
                text += " "
            label = int(line[0]) - 1
        
            documents.append(text)
            labels.append(label)

    return documents, labels

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # get model HAN
    model = HANnet(args['word_hidden_size'], args['sent_hidden_size'], 1, args['word2vec_path'], 4)
    model = load_model(args['pre_trained_model'], model, device)
    
    documents, labels = get_test_documents(args['data_path'])
    a_doc = documents[11]
    print('e.g: ', a_doc)
    a_label = labels[11]
    
    max_length_word, max_length_sent = get_max_lengths(args['data_path'])

    feature = doc_tokenizer(a_doc, args['word2vec_path'], max_length_sent, max_length_word)
    print('shape: ', feature.shape)
    labels = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    prob, label = predict(feature, model, labels)
    print('Prediction: prob %f , label %s ' % (prob.item(), label))
    print('True: %s'%a_label)

    