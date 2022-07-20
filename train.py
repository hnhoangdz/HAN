import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths
from src.dataset import MyDataset
from src.HAN_net import HANnet
import argparse
import shutil
import numpy as np
from tqdm import tqdm, trange
from collections import Counter
import time


def get_args():

    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str,
                        default="data/ag_news_csv/train.csv")
    parser.add_argument("--test_set", type=str,
                        default="data/ag_news_csv/test.csv")
    parser.add_argument("--test_interval", type=int, default=1,
                        help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str,
                        default="data/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = vars(parser.parse_args())

    return args


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def dataloader(params):

    training_params = {"batch_size": params['batch_size'],
                       "shuffle": True,
                       "drop_last": True}

    test_params = {"batch_size": params['batch_size'],
                   "shuffle": False,
                   "drop_last": False}

    max_word_length, max_sent_length = get_max_lengths(params['train_set'])

    train_set = MyDataset(
        params['train_set'], params['word2vec_path'], max_sent_length, max_word_length)
    train_dataloader = DataLoader(train_set, **training_params)

    test_set = MyDataset(
        params['test_set'], params['word2vec_path'], max_sent_length, max_word_length)
    test_dataloader = DataLoader(test_set, **test_params)

    return train_dataloader, test_dataloader


def train(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()

    for (x, y) in tqdm(data_loader, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        model._init_hidden_state()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def evaluate(model, data_loader, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(data_loader, desc="Evaluating", leave=False):

            num_samples = len(y)
            x = x.to(device)
            y = y.to(device)

            model._init_hidden_state(num_samples)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def train_process(model, optimizer, criterion, scheduler, num_epochs, weight_path='trained_models/best_model'):
    # store best test loss
    best_test_loss = float('inf')

    # training process
    for epoch in trange(num_epochs, desc="Epochs"):

        start_time = time.monotonic()

        train_loss, train_acc = train(
            model, train_dataloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(
            model, test_dataloader, criterion, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion
                        }, weight_path + '_' + str(epoch) + '_' + str(test_loss) + '.pth')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        scheduler.step(test_loss)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')


if __name__ == '__main__':
    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialization in random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # command line input parameters
    args = get_args()

    # number of epochs
    EPOCHS = args['num_epoches']

    # number of classes
    num_classes = 4

    # data loader
    train_dataloader, test_dataloader = dataloader(args)

    # load model
    model = HANnet(args['word_hidden_size'], args['sent_hidden_size'],
                   args['batch_size'], args['word2vec_path'], num_classes)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    # criterion
    criterion = nn.CrossEntropyLoss()

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args['es_patience'])

    # Training
    train_process(model, optimizer, criterion, scheduler, EPOCHS)
