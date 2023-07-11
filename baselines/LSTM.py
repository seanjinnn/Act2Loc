import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class LSTM(nn.Module):
    def __init__(self, dataset):
        super(LSTM, self).__init__()
        self.loc_dim = 256
        self.hidden_size = 1024
        self.num_layers = 3

        # 初始化嵌入矩阵
        self.emb_loc = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.loc_dim)

        self.lstm = nn.LSTM(
            input_size=self.loc_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.fc = nn.Linear(self.hidden_size, n_vocab)

    def forward(self, loc, prev_state):
        # 拼接嵌入矩阵
        loc_emb = torch.squeeze(self.emb_loc(loc))

        output, state = self.lstm(loc_emb, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size))


class Dataset():
    def __init__(self, dataset):
        self.args = args
        self.words = np.ravel(np.array(dataset))
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [[self.word_to_index[w] for w in activity] for activity in dataset]

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        """返回数据集当中的样本个数"""
        return len(self.words_indexes)

    def __getitem__(self, index):
        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        return (torch.tensor(self.words_indexes[index][:-1]),
                torch.tensor(self.words_indexes[index][1:]))


def train(dataset, time, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})

def predict(dataset, time, model, text, next_words=168):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    new_words = []
    with torch.no_grad():
        for batch, (x, y) in tqdm(enumerate(dataloader)):
            state_h, state_c = model.init_state(args.sequence_length)
            state_h = state_h.to(device)
            state_c = state_c.to(device)

            new_word = x.cpu()
            for i in range(next_words):
                y_pred, (state_h, state_c) = model(torch.tensor(new_word[:, i:].to(device)), (state_h, state_c))

                col = np.array([], dtype=int)
                for row in range(len(y_pred)):
                    last_word_logits = y_pred[row][-1]

                    p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
                    word_index = np.random.choice(len(last_word_logits), p=p)
                    col = np.append(col, word_index)

                if i == 0:
                    predict = col[:, np.newaxis]
                else:
                    predict = np.concatenate((predict, col[:, np.newaxis]), axis=1)

                col = torch.tensor(col[:, np.newaxis])
                new_word = torch.cat((new_word, col), dim=1)

            new_words.extend(predict.tolist())

    return new_words