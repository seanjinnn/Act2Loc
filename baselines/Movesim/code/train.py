# encoding: utf-8
import torch
import numpy as np
import time
from tqdm import tqdm
def generate_samples(model, batch_size, seq_len, generated_num, output_file):
    samples = []
    for _ in tqdm(range(int(generated_num / batch_size))):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def generate_samples_to_mem(model, batch_size, seq_len, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)


def pretrain_model(
        name,
        pre_epochs,
        model,
        data_iter,
        criterion,
        optimizer,
        batch_size,
        device=None):
    lloss = 0.
    criterion = criterion.to(device)
    for epoch in tqdm(range(pre_epochs)):
        stime = time.time()
        loss = train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device)
        etime = time.time()
        epoch_time = (etime - stime) / 60
        print('Epoch [%d], loss: %f' % (epoch + 1, loss))
        # print('Epoch Time: {}'.format(epoch_time))
        if loss < 0.01 or 0 < lloss - loss < 0.01:
            print("early stop at epoch %d" % (epoch + 1))
            break

def train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device=None):
    total_loss = 0.
    if name == "G":
        tim = torch.LongTensor([i%24 for i in range(167)]).to(device)
        # 47 = 24 * 2 - 1，减1是因为最后一个时间点不需要输入
        # 最后一个时间点不需要输入的原因是，最后一个时间点的输入是第一个时间点的输出
        tim = tim.repeat(batch_size).reshape(batch_size, -1)        # (batch_size, 47)
    start_time = time.time()
    for i, (data, target) in tqdm(enumerate(data_iter)):
        data = torch.LongTensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        target = target.contiguous().view(-1)
        if name == "G":
            pred = model(data, tim)
        else:
            pred = model(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    end_time = time.time()
    epoch_time = (end_time - start_time) / 60
    print("epoch time: %f" % epoch_time)
    return total_loss / (i + 1)
