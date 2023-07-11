import argparse
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data_iter import DisDataIter, GenDataIter
from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from loss import PGLoss

# Arguemnts-arg
'''
ArgumentParser 对象
    prog - 程序的名称（默认：sys.argv[0]）
    usage - 描述程序用途的字符串（默认值：从添加到解析器的参数生成）
    description - 在参数帮助文档之前显示的文本（默认值：无）
    epilog - 在参数帮助文档之后显示的文本（默认值：无）
    parents - 一个 ArgumentParser 对象的列表，它们的参数也应包含在内
    formatter_class - 用于自定义帮助文档输出格式的类
    prefix_chars - 可选参数的前缀字符集合（默认值：’-’）
    fromfile_prefix_chars - 当需要从文件中读取其他参数时，用于标识文件名的前缀字符集合（默认值：None）
    argument_default - 参数的全局默认值（默认值： None）
    conflict_handler - 解决冲突选项的策略（通常是不必要的）
    add_help - 为解析器添加一个 -h/–help 选项（默认值： True）
    allow_abbrev - 如果缩写是无歧义的，则允许缩写长选项 （默认值：True）

add_argument() 方法
    name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
    action - 当参数在命令行中出现时使用的动作基本类型。
    nargs - 命令行参数应当消耗的数目。
    const - 被一些 action 和 nargs 选择所需求的常数。
    default - 当参数未在命令行中出现时使用的值。
    type - 命令行参数应当被转换成的类型。
    choices - 可用的参数的容器。
    required - 此命令行选项是否可省略 （仅选项可用）。
    help - 一个此选项作用的简单描述。
    metavar - 在使用方法消息中使用的参数值示例。
    dest - 被添加到 parse_args() 所返回对象上的属性名
'''
parser = argparse.ArgumentParser(description='SeqGAN')      # description - 在参数帮助文档之前显示的文本（默认值：无）
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data_path', type=str, default='', metavar='PATH',
                    help='data path to save files (default: /scratch/zc807/seq_gan/)')
parser.add_argument('--rounds', type=int, default=15, metavar='N',
                    help='rounds of adversarial training (default: 150)')
parser.add_argument('--g_pretrain_steps', type=int, default=120, metavar='N',
                    help='steps of pre-training of generators (default: 120)')
parser.add_argument('--d_pretrain_steps', type=int, default=50, metavar='N',
                    help='steps of pre-training of discriminators (default: 50)')
parser.add_argument('--g_steps', type=int, default=1, metavar='N',
                    help='steps of generator updates in one round of adversarial training (default: 1)')
parser.add_argument('--d_steps', type=int, default=3, metavar='N',
                    help='steps of discriminator updates in one round of adversarial training (default: 3)')
parser.add_argument('--gk_epochs', type=int, default=1, metavar='N',
                    help='epochs of generator updates in one step of generate update (default: 1)')
parser.add_argument('--dk_epochs', type=int, default=3, metavar='N',
                    help='epochs of discriminator updates in one step of discriminator update (default: 3)')
parser.add_argument('--update_rate', type=float, default=0.8, metavar='UR',
                    help='update rate of roll-out model (default: 0.8)')
parser.add_argument('--n_rollout', type=int, default=16, metavar='N',
                    help='number of roll-out (default: 16)')
parser.add_argument('--vocab_size', type=int, default=2236, metavar='N',
                    help='vocabulary size (default: 14)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--n_samples', type=int, default=200000, metavar='N',
                    help='number of samples gerenated per time (default: 6400)')
parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 1e-3)')
parser.add_argument('--dis_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of discriminator optimizer (default: 1e-3)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Files
POSITIVE_FILE = 'sz_1km.data'
NEGATIVE_FILE = 'gene.data'

# Genrator Parameters
g_embed_dim = 32
g_hidden_dim = 32
g_seq_len = 168


# Discriminator Parameters
d_num_class = 2
d_embed_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]            # 12
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout_prob = 0.2


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []

    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)

    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('{}\n'.format(string))

def generate_sampless(model, batch_size, generated_num, output_file, epoch):
    samples = []

    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)

    with open(output_file+str(epoch), 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('{}\n'.format(string))

def train_generator_MLE(gen, data_iter, criterion, optimizer, epochs, 
        gen_pretrain_train_loss, args):
    """
    Train generator with MLE
    """
    for epoch in range(epochs):
        total_loss = 0.
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)           # target的维度为[batch_size * seq_len]
            output = gen(data)                              # output的维度为[batch_size * seq_len, vocab_size]
            loss = criterion(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_iter.reset()
    avg_loss = total_loss / len(data_iter)
    print("Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))
    gen_pretrain_train_loss.append(avg_loss)

def train_generator_PG(gen, dis, rollout, pg_loss, optimizer, epochs, args):
    """
    Train generator with the guidance of policy gradient
    """
    for epoch in range(epochs):
        # construct the input to the genrator, add zeros before samples and delete the last column
        samples = generator.sample(args.batch_size, g_seq_len)
        zeros = torch.zeros(args.batch_size, 1, dtype=torch.int64)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous()     # inputs的维度为[batch_size, seq_len]
        targets = samples.data.contiguous().view((-1,))                     # targets的维度为[batch_size * seq_len]

        # calculate the reward
        rewards = torch.tensor(rollout.get_reward(samples, args.n_rollout, dis))
        if args.cuda:
            rewards = rewards.cuda()

        # update generator
        output = gen(inputs)
        loss = pg_loss(output, targets, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_generator(model, data_iter, criterion, args):
    """
    Evaluate generator with NLL
    """
    total_loss = 0.
    with torch.no_grad():
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    return avg_loss


def train_discriminator(dis, gen, criterion, optimizer, epochs, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Train discriminator
    """
    generate_samples(gen, args.batch_size, args.n_samples, NEGATIVE_FILE)
    data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
    for epoch in range(epochs):
        correct = 0
        total_loss = 0.
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = dis(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = criterion(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_iter.reset()
        avg_loss = total_loss / len(data_iter)
        acc = correct.item() / data_iter.data_num
        print("Epoch {}, train loss: {:.5f}, train acc: {:.3f}".format(epoch, avg_loss, acc))
        dis_adversarial_train_loss.append(avg_loss)
        dis_adversarial_train_acc.append(acc)


def eval_discriminator(model, data_iter, criterion, args):
    """
    Evaluate discriminator, dropout is enabled
    """
    correct = 0
    total_loss = 0.
    with torch.no_grad():
        for data, target in data_iter:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    acc = correct.item() / data_iter.data_num
    return avg_loss, acc


def adversarial_train(gen, dis, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Adversarially train generator and discriminator
    """
    # train generator for g_steps
    print("#Train generator")
    for i in range(args.g_steps):
        print("##G-Step {}".format(i))
        train_generator_PG(gen, dis, rollout, pg_loss, gen_optimizer, args.gk_epochs, args)

    # train discriminator for d_steps
    print("#Train discriminator")
    for i in range(args.d_steps):
        print("##D-Step {}".format(i))
        train_discriminator(dis, gen, nll_loss, dis_optimizer, args.dk_epochs, 
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)

    # update roll-out model
    rollout.update_params()


if __name__ == '__main__':
    # Parse arguments，解析参数
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if not args.hpc:
        args.data_path = ''
    POSITIVE_FILE = args.data_path + POSITIVE_FILE
    NEGATIVE_FILE = args.data_path + NEGATIVE_FILE

    # Set models, criteria, optimizers
    # models
    generator_dict = torch.load('save_model/pretrain_generator.pth')
    generator = Generator(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda)
    generator.load_state_dict(generator_dict)

    discriminator_dict = torch.load('save_model/pretrain_discriminator.pth')
    discriminator = nn.DataParallel(Discriminator(d_num_class, args.vocab_size, d_embed_dim, d_filter_sizes, d_num_filters, d_dropout_prob))
    discriminator.load_state_dict(discriminator_dict)
    target_lstm = nn.DataParallel(TargetLSTM(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda))

    # criterion
    nll_loss = nn.NLLLoss()
    pg_loss = PGLoss()              # policy gradient Loss, for adversarial training of Generator
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        target_lstm = target_lstm.cuda()
        nll_loss = nll_loss.cuda()
        pg_loss = pg_loss.cuda()
        cudnn.benchmark = True

    print("criterion")

    # optimizer
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=args.gen_lr)
    dis_optimizer = optim.SGD(params=discriminator.parameters(), lr=args.dis_lr)

    # Container of experiment data
    gen_pretrain_train_loss = []                # 生成器预训练训练集损失
    gen_pretrain_eval_loss = []                 # 生成器预训练测试集损失
    dis_pretrain_train_loss = []                # 辨别器预训练训练集损失
    dis_pretrain_train_acc = []                 # 辨别器预训练训练正确性
    dis_pretrain_eval_loss = []                 # 辨别器预训练测试集损失
    dis_pretrain_eval_acc = []                  # 辨别器预训练测试正确性
    gen_adversarial_eval_loss = []              # 生成器对抗训练测试集损失
    dis_adversarial_train_loss = []             # 辨别器对抗训练训练集损失
    dis_adversarial_train_acc = []              # 辨别器对抗训练训练正确性
    dis_adversarial_eval_loss = []              # 辨别器对抗训练测试集损失
    dis_adversarial_eval_acc = []               # 辨别器对抗训练测试正确性

    # Generate toy data using target LSTM
    # print('#####################################################')
    # print('Generating data ...')
    # print('#####################################################\n\n')
    # generate_samples(target_lstm, args.batch_size, args.n_samples, POSITIVE_FILE)

    '''
    tf_预训练Generator
    我们首先定义了预训练过程中Generator的优化器，即通过AdamOptimizer来最小化交叉熵损失，
    随后我们通过target-lstm网络来产生Generator的训练数据，利用dataloader来读取每一个batch的数据。
    同时，每隔一定的步数，我们会计算Generator与target-lstm的相似性(likelihood)
    '''
    # # # Pre-train generator using MLE
    # print('#####################################################')
    # print('Start pre-training generator with MLE...')
    # print('#####################################################\n')
    # gen_data_iter = GenDataIter(POSITIVE_FILE, args.batch_size)
    # for i in range(args.g_pretrain_steps):
    #     print("G-Step {}".format(i))
    #     train_generator_MLE(generator, gen_data_iter, nll_loss,
    #         gen_optimizer, args.gk_epochs, gen_pretrain_train_loss, args)
    #
    #     generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)
    #
    #     eval_iter = GenDataIter(NEGATIVE_FILE, args.batch_size)
    #     gen_loss = eval_generator(target_lstm, eval_iter, nll_loss, args)
    #     gen_pretrain_eval_loss.append(gen_loss)
    #     print("eval loss: {:.5f}\n".format(gen_loss))
    #
    # # # 保存模型
    # torch.save(generator.state_dict(), 'save_model/pretrain_generator.pth')
    # print('#####################################################\n\n')

    '''
    tf.预训练Discriminator
    预训练好Generator之后，我们就可以通过Generator得到一批负样本，并结合target-lstm产生的正样本来预训练我们的Discriminator。
    '''
    # # # Pre-train discriminator
    # print('#####################################################')
    # print('Start pre-training discriminator...')
    # print('#####################################################\n')
    # for i in range(args.d_pretrain_steps):
    #     print("D-Step {}".format(i))
    #     train_discriminator(discriminator, generator, nll_loss,
    #         dis_optimizer, args.dk_epochs, dis_adversarial_train_loss, dis_adversarial_train_acc, args)
    #     generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)
    #     eval_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
    #     dis_loss, dis_acc = eval_discriminator(discriminator, eval_iter, nll_loss, args)
    #     dis_pretrain_eval_loss.append(dis_loss)
    #     dis_pretrain_eval_acc.append(dis_acc)
    #     print("eval loss: {:.5f}, eval acc: {:.3f}\n".format(dis_loss, dis_acc))
    #
    # # 保存模型
    # torch.save(discriminator.state_dict(), 'save_model/pretrain_discriminator.pth')
    # print('#####################################################\n\n')


    '''
    tf.
    对抗过程中训练Generator，我们首先需要通过Generator得到一批序列sample，
    然后使用roll-out结合Discriminator得到每个序列中每个时点的reward，
    再将reward和sample喂给adversarial_network进行参数更新。
    对抗过程中Discriminator的训练和预训练过程一样.
    '''
    # Adversarial training
    print('#####################################################')
    print('Start adversarial training...')
    print('#####################################################\n')
    rollout = Rollout(generator, args.update_rate)
    for i in range(args.rounds):
        print("Round {}".format(i))
        adversarial_train(generator, discriminator, rollout,
            pg_loss, nll_loss, gen_optimizer, dis_optimizer,
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)
        generate_samples(generator, args.batch_size, args.n_samples, NEGATIVE_FILE)
        if i == args.rounds - 1:
            for epoch in range(9):
                generate_sampless(generator, args.batch_size, args.n_samples, NEGATIVE_FILE, epoch)
        gen_eval_iter = GenDataIter(NEGATIVE_FILE, args.batch_size)
        dis_eval_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, args.batch_size)
        gen_loss = eval_generator(target_lstm, gen_eval_iter, nll_loss, args)
        gen_adversarial_eval_loss.append(gen_loss)
        dis_loss, dis_acc = eval_discriminator(discriminator, dis_eval_iter, nll_loss, args)
        dis_adversarial_eval_loss.append(dis_loss)
        dis_adversarial_eval_acc.append(dis_acc)
        print("gen eval loss: {:.5f}, dis eval loss: {:.5f}, dis eval acc: {:.3f}\n"
            .format(gen_loss, dis_loss, dis_acc))

    # Save experiment data
    with open(args.data_path + 'experiment.pkl', 'wb') as f:
        pkl.dump(
            (gen_pretrain_train_loss,
                gen_pretrain_eval_loss,
                dis_pretrain_train_loss,
                dis_pretrain_train_acc,
                dis_pretrain_eval_loss,
                dis_pretrain_eval_acc,
                gen_adversarial_eval_loss,
                dis_adversarial_train_loss,
                dis_adversarial_train_acc,
                dis_adversarial_eval_loss,
                dis_adversarial_eval_acc),
            f,
            protocol=pkl.HIGHEST_PROTOCOL
        )
