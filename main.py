# -*- coding: utf-8 -*-
import time
import argparse
import numpy as np
import random, os

import torch
import torch.nn.functional as F
import torch.optim as optim

from strnn import STRNN
from util import loadData, convert_to_one_hot, evaluation_4class, loadData_Cross
from sklearn.metrics import accuracy_score, classification_report

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='twitter15',
                    help='the dataset name: twitter15, twitter16(default: twitter15)')
parser.add_argument('--vocab_size', type=int, default=5000,
                    help='the size of vocabulary (default:5000s)')
parser.add_argument('--input_size', type=int, default=300,
                    help='the dimension of word embedding (default: 300)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='Number of hidden units.')
parser.add_argument('--elapsed_time',type=int, default=3000,
                    help='the elapsed time after source tweet posted (0, 60(1h), 120(2h), 240(4h), 480(8h), 720(12h), 1440(24h), 2160(36h), default: 3000 represents all)')
parser.add_argument('--tweets_count', type=int, default=500,
                    help='the tweets count after source tweet posted (0, 10, 20, 40, 60, 80, 200, 300, default: 500 represents all)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--filename', type = str, default = "",
                                    help='output file')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
target_names = ['NR','FR','TR','UR']
if args.dataset == 'weibo':
    target_names = ['NR','FR']

random.seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(epoch, best_acc, patience, fold_id):
    model.train()
    total_iters = len(sequences_train)//args.batch_size + 1
    loss_accum = 0
    avg_acc = 0
    idx_list = np.arange(len(sequences_train))
    random.shuffle(idx_list)
    for i in range(total_iters):
        selected_idx = idx_list[(i*args.batch_size):((i+1)*args.batch_size)]
        if len(selected_idx) == 0:
            continue
        output = []
        loss_train = []
        for j in selected_idx:
            x_index = index_train[j]
            sequence = sequences_train[j]
            out = model(x_index, sequence)
            y = y_train[j].view(-1)
            # print(y.size())
            loss = F.nll_loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output.append(out)
            loss_train.append(round(float(loss.detach().cpu().numpy()),2))
        output = torch.cat([out for out in output], 0)
        y_selected = y_train[selected_idx]

        corrects = (torch.max(output, 1)[1].view(len(selected_idx)).data == y_selected.data).sum()
        accuracy = 100*corrects/len(selected_idx)

        if i > 0 and i % 100 == 0:
            best_acc, patience = evaluate(best_acc, patience, fold_id)
            model.train()
        avg_acc += accuracy
        print('Batch [{}] - loss:{:.6f} acc:{:.4f}% ({}/{})'.format(i, np.mean(loss_train), accuracy, corrects, len(selected_idx)))
        loss_accum += np.mean(loss_train)
    average_loss = loss_accum/total_iters
    average_acc = avg_acc/total_iters
    print("loss training: {:.6f} average_acc: {:.6f}".format(average_loss, average_acc))

    return best_acc, patience

def adjust_learning_rate(optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr

def predict(index_list, sequences_list):
    model.eval()
    output = []
    for i,x_index in enumerate(index_list):
        sequence = sequences_list[i]
        out = model(x_index, sequence)
        output.append(out)

    output = torch.cat([out for out in output], 0)
    return output

def evaluate(best_acc, patience, fold_id):
    output = predict(index_dev, sequences_dev)
    predicted = torch.max(output, dim=1)[1]
    y_pred = predicted.data.cpu().numpy().tolist()
    val_labels = y_dev.data.cpu().numpy().tolist()
    acc = accuracy_score(val_labels, y_pred)

    if acc > best_acc:
        best_acc = acc
        patience = 0
        if args.elapsed_time == 3000 and args.tweets_count == 500:
            torch.save(model.state_dict(), 'weights.best.{}.f{}'.format(args.dataset, fold_id))
        elif args.elapsed_time < 3000 and args.tweets_count == 500:
            torch.save(model.state_dict(), 'weights.best.{}.f{}.et{}'.format(args.dataset,fold_id, args.elapsed_time))
        elif args.elapsed_time == 3000 and args.tweets_count < 500:
            torch.save(model.state_dict(), 'weights.best.{}.f{}.tc{}'.format(args.dataset, fold_id, args.tweets_count))
        print(classification_report(val_labels, y_pred, target_names=target_names, digits=5))
        print('Val set acc: {}'.format(acc))
        print('Best val set acc: {}'.format(best_acc))
        print('save model!!!!')
    else:
        patience += 1

    return best_acc, patience

def test(fold_id):
    # model.eval()
    output = predict(index_test, sequences_test)
    predicted = torch.max(output, dim=1)[1]
    y_pred = predicted.data.cpu().numpy().tolist()
    test_labels = y_test.data.cpu().numpy().tolist()
    print('=====================================')
    print('the result of {}-fold:'.format(fold_id))
    print(classification_report(test_labels, y_pred, target_names=target_names, digits=5))
    with open('result.{}.f{}.et{}.tc{}'.format(args.dataset, fold_id, args.elapsed_time, args.tweets_count), 'w') as f:
        f.write('====================================='+'\n')
        f.write(classification_report(test_labels, y_pred, target_names=target_names, digits=5))
    t_labels = convert_to_one_hot(y_test.unsqueeze(1).cpu(), 4).cuda()
    if args.dataset == 'weibo':
        t_labels = convert_to_one_hot(y_test.unsqueeze(1).cpu(), 2).cuda()
    result_test = evaluation_4class(output, t_labels)
    return result_test

fold_ids = [i for i in range(5)]
result_test_dict = {}
for i in range(20):
    result_test_dict[i] = []
for fold_id in fold_ids:
    sequences_train, index_train, y_train, sequences_dev, index_dev, y_dev, sequences_test, index_test, y_test = loadData_Cross(args.dataset, args.vocab_size, args.elapsed_time, args.tweets_count, fold_id)

    model = STRNN(vocab_size=args.vocab_size,
                  input_size=args.input_size,
                  hidden_size=args.hidden_size,
                  nclass=y_train.max().item() + 1,
                  device=args.cuda)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        y_train = y_train.cuda()
        y_dev = y_dev.cuda()
        y_test = y_test.cuda()

    # Train model
    best_acc = 0.0
    patience = 0
    t_total = time.time()
    for epoch in range(1, args.epochs+1):
        print("Epoch {}/{}".format(epoch, args.epochs))
        best_acc, patience = train(epoch, best_acc, patience,fold_id)
        if epoch >= 10 and patience > 3:
            print('Reload the best model ...')
            if args.elapsed_time == 3000 and args.tweets_count == 500:
                model.load_state_dict(torch.load('weights.best.{}.f{}'.format(args.dataset, fold_id)))
            elif args.elapsed_time < 3000 and args.tweets_count == 500:
                model.load_state_dict(torch.load('weights.best.{}.f{}.et{}'.format(args.dataset, fold_id, args.elapsed_time)))
            elif args.elapsed_time == 3000 and args.tweets_count < 500:
                model.load_state_dict(torch.load('weights.best.{}.f{}.tc{}'.format(args.dataset, fold_id, args.tweets_count)))
            now_lr = adjust_learning_rate(optimizer)
            print(now_lr)
            patience = 0
        best_acc, patience = evaluate(best_acc, patience, fold_id)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    print('Loading model to test set ...')
    if args.elapsed_time == 3000 and args.tweets_count == 500:
        model.load_state_dict(torch.load('weights.best.{}.f{}'.format(args.dataset, fold_id)))
    elif args.elapsed_time < 3000 and args.tweets_count == 500:
        model.load_state_dict(torch.load('weights.best.{}.f{}.et{}'.format(args.dataset, fold_id, args.elapsed_time)))
    elif args.elapsed_time == 3000 and args.tweets_count < 500:
        model.load_state_dict(torch.load('weights.best.{}.f{}.tc{}'.format(args.dataset, fold_id, args.tweets_count)))
    result_test = test(fold_id)
    for i in range(20):
        result_test_dict[i].append(result_test[i])

result = []
for i in range(20):
    result.append(np.mean(result_test_dict[i]))
print('the result of {}-fold cross validation in test set:'.format(len(fold_ids)))
print('acc:{:.4f} Favg:{:.4f},{:.4f},{:.4f}'.format(result[0], result[1], result[2], result[3]) +
        ' C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[4], result[5], result[6], result[7]) +
        ' C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[8], result[9], result[10], result[11]) +
        ' C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[12], result[13], result[14], result[15]) +
        ' C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[16], result[17], result[18], result[19]))
if not args.filename == "":
    with open(args.filename, 'w') as f:
        f.write('the result of {}-fold cross validation in test set:'.format(len(fold_ids)))
        f.write('acc:{:.4f} Favg:{:.4f},{:.4f},{:.4f}'.format(result[0], result[1], result[2], result[3]) +
                ' C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[4], result[5], result[6], result[7]) +
                ' C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[8], result[9], result[10], result[11]) +
                ' C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[12], result[13], result[14], result[15]) +
                ' C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result[16], result[17], result[18], result[19]))
