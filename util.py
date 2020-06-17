# -*- coding: utf-8 -*-
import torch

import codecs
import logging,sys,os
#logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       # y_train = [1,0,0,0]
       y_train = 0
       l1 += 1
    if label in labelset_f:
       # y_train = [0,1,0,0]
       y_train = 1
       l2 += 1
    if label in labelset_t:
       # y_train = [0,0,1,0]
       y_train = 2
       l3 += 1
    if label in labelset_u:
       # y_train = [0,0,0,1]
       y_train = 3
       l4 += 1
    return y_train, l1,l2,l3,l4

def padded_index(idx_str, maxL, vocab_size):
    wordIndex = []
    l = 0
    for idx in idx_str.split(' '):
        wordIndex.append(int(idx))
        l += 1
    padded = [(vocab_size-1) for i in range(maxL-l)]
    wordIndex += padded
    return wordIndex

def loadData(dataset, vocab_size, elapsed_time, tweets_count):
    labels_path = 'dataset/{}/label.txt'.format(dataset)
    logger.info("loading source message label")
    labelDict = {}
    for line in open(labels_path):
        line = line.strip()
        label, eid = line.split(':')[0], line.split(':')[1]
        labelDict[eid] = label.lower()
    logger.info(len(labelDict))

    logger.info('reading sequences')
    if elapsed_time == 3000 and tweets_count == 500:
        sequences_path = 'dataset/{}/data.strnn.vol{}.txt'.format(dataset, vocab_size)
    elif elapsed_time != 3000:
        sequences_path = 'dataset/{}/data.strnn.vol{}.et{}.txt'.format(dataset, vocab_size, elapsed_time)
    elif tweets_count != 500:
        sequences_path = 'dataset/{}/data.strnn.vol{}.tc{}.txt'.format(dataset, vocab_size, tweets_count)
    sequencesDict = {}
    wordidxDict = {}
    for line in open(sequences_path):
        line = line.strip()
        node = []
        eid, indexC = line.split('\t')[0], line.split('\t')[1]
        node.append(int(indexC))

        indexParents, indexPeriors = line.split('\t')[2], line.split('\t')[3]
        parents = []
        # print('eid:{}'.format(eid))
        for idx in indexParents.split(' '):
            if idx == 'None':
                parents.append(-1)
            else:
                parents.append(int(idx))
        node.append(parents)
        perior_times = []
        for idx in indexPeriors.split(' '):
            if idx == 'None':
                perior_times.append(-1)
            else:
                perior_times.append(int(idx))
        node.append(perior_times)

        maxL = int(line.split('\t')[4])
        word_index_str = line.split('\t')[5]
        # padded_word_index = padded_index(word_index_str, maxL)
        if eid not in wordidxDict:
            wordidxDict[eid] = []
        #wordidxDict[eid].append(padded_word_index)
        wordidxDict[eid].append([int(idx) for idx in word_index_str.split(' ')])

        if eid not in sequencesDict:
            sequencesDict[eid] = []
        sequencesDict[eid].append(node)
    logger.info('tree no:{}'.format(len(sequencesDict)))

    logger.info('loading train set')
    trainPath = 'dataset/{}/{}.train'.format(dataset, dataset)
    sequences_train, index_train, y_train = [], [], []
    l1,l2,l3,l4 = 0,0,0,0
    for line in open(trainPath):
        line = line.strip()
        eid = line.split('\t')[0]
        if eid not in labelDict: continue
        if eid not in sequencesDict: continue
        if len(sequencesDict[eid]) <= 0:
            continue
        label = labelDict[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        sequences_train.append(sequencesDict[eid])
        index_train.append(wordidxDict[eid])
    logger.info('train set:{} non-rumor, {} false-rumor, {} true-rumor, {} un-rumor'.format(l1,l2,l3,l4))

    logger.info('loading dev set')
    devPath = 'dataset/{}/{}.dev'.format(dataset, dataset)
    sequences_dev, index_dev, y_dev = [], [], []
    l1,l2,l3,l4 = 0,0,0,0
    for line in open(devPath):
        line = line.strip()
        eid = line.split('\t')[0]
        if eid not in labelDict: continue
        if eid not in sequencesDict: continue
        if len(sequencesDict[eid]) <= 0:
            continue
        label = labelDict[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_dev.append(y)
        sequences_dev.append(sequencesDict[eid])
        index_dev.append(wordidxDict[eid])
    logger.info('dev set:{} non-rumor, {} false-rumor, {} true-rumor, {} un-rumor'.format(l1,l2,l3,l4))

    logger.info('loading test set')
    testPath = 'dataset/{}/{}.test'.format(dataset, dataset)
    sequences_test, index_test, y_test = [], [], []
    l1,l2,l3,l4 = 0,0,0,0
    for line in open(testPath):
        line = line.strip()
        eid = line.split('\t')[0]
        if eid not in labelDict: continue
        if eid not in sequencesDict: continue
        if len(sequencesDict[eid]) <= 0:
            continue
        label = labelDict[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        sequences_test.append(sequencesDict[eid])
        index_test.append(wordidxDict[eid])
    logger.info('test set:{} non-rumor, {} false-rumor, {} true-rumor, {} un-rumor'.format(l1,l2,l3,l4))

    y_train = torch.LongTensor(y_train)
    y_dev = torch.LongTensor(y_dev)
    y_test = torch.LongTensor(y_test)

    return sequences_train, index_train, y_train, sequences_dev, index_dev, y_dev, sequences_test, index_test, y_test

def loadData_Cross(dataset, vocab_size, elapsed_time, tweets_count, fold):
    labels_path = 'dataset/{}/label.txt'.format(dataset)
    logger.info("loading source message label")
    labelDict = {}
    for line in open(labels_path):
        line = line.strip()
        label, eid = line.split(':')[0], line.split(':')[1]
        labelDict[eid] = label.lower()
    logger.info(len(labelDict))

    logger.info('reading sequences')
    if elapsed_time == 3000 and tweets_count == 500:
        sequences_path = 'dataset/{}/data.strnn.vol{}.txt'.format(dataset, vocab_size)
    elif elapsed_time != 3000:
        sequences_path = 'dataset/{}/data.strnn.vol{}.f{}.et{}.txt'.format(dataset, vocab_size, fold, elapsed_time)
    elif tweets_count != 500:
        sequences_path = 'dataset/{}/data.strnn.vol{}.f{}.tc{}.txt'.format(dataset, vocab_size, fold, tweets_count)
    sequencesDict = {}
    wordidxDict = {}
    for line in open(sequences_path):
        line = line.strip()
        node = []
        eid, indexC = line.split('\t')[0], line.split('\t')[1]
        node.append(int(indexC))

        indexParents, indexPeriors = line.split('\t')[2], line.split('\t')[3]
        parents = []
        # print('eid:{}'.format(eid))
        for idx in indexParents.split(' '):
            if idx == 'None':
                parents.append(-1)
            else:
                parents.append(int(idx))
        node.append(parents)
        perior_times = []
        for idx in indexPeriors.split(' '):
            if idx == 'None':
                perior_times.append(-1)
            else:
                perior_times.append(int(idx))
        node.append(perior_times)

        maxL = int(line.split('\t')[4])
        word_index_str = line.split('\t')[5]
        # padded_word_index = padded_index(word_index_str, maxL)
        if eid not in wordidxDict:
            wordidxDict[eid] = []
        #wordidxDict[eid].append(padded_word_index)
        wordidxDict[eid].append([int(idx) for idx in word_index_str.split(' ')])

        if eid not in sequencesDict:
            sequencesDict[eid] = []
        sequencesDict[eid].append(node)
    logger.info('tree no:{}'.format(len(sequencesDict)))

    logger.info('loading train set')
    trainPath = 'dataset/nfold/RNNtrainSet_{}{}_tree.txt'.format(dataset, fold)
    eid_list = []
    for line in open(trainPath):
        eid_list.append(line.strip())

    non_eid_list = []
    false_eid_list = []
    true_eid_list = []
    un_eid_list = []
    for eid in eid_list:
        if labelDict[eid] in ['news', 'non-rumor']:
            non_eid_list.append(eid)
        if labelDict[eid] in ['false']:
            false_eid_list.append(eid)
        if labelDict[eid] in ['true']:
            true_eid_list.append(eid)
        if labelDict[eid] in ['unverified']:
            un_eid_list.append(eid)

    split_index = int(len(sequencesDict)*0.1)//4

    # non_index = len(non_eid_list)//4
    # false_index = len(false_eid_list)//4
    # true_index = len(true_eid_list)//4
    # un_index = len(un_eid_list)//4

    train_ids = non_eid_list[:-split_index] + false_eid_list[:-split_index] + true_eid_list[:-split_index] + un_eid_list[:-split_index]
    dev_ids = non_eid_list[-split_index:] + false_eid_list[-split_index:] + true_eid_list[-split_index:] + un_eid_list[-split_index:]

    sequences_train, index_train, y_train = [], [], []
    l1,l2,l3,l4 = 0,0,0,0
    # for line in open(trainPath):
    #     eid = line.strip()
    for eid in train_ids:
        if eid not in labelDict: continue
        if eid not in sequencesDict: continue
        if len(sequencesDict[eid]) <= 0:
            continue
        label = labelDict[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        sequences_train.append(sequencesDict[eid])
        index_train.append(wordidxDict[eid])
    logger.info('train set:{} non-rumor, {} false-rumor, {} true-rumor, {} un-rumor'.format(l1,l2,l3,l4))

    logger.info('loading dev set')
    # devPath = 'dataset/{}/{}.dev'.format(dataset, dataset)
    sequences_dev, index_dev, y_dev = [], [], []
    l1,l2,l3,l4 = 0,0,0,0
    for eid in dev_ids:
        if eid not in labelDict: continue
        if eid not in sequencesDict: continue
        if len(sequencesDict[eid]) <= 0:
            continue
        label = labelDict[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_dev.append(y)
        sequences_dev.append(sequencesDict[eid])
        index_dev.append(wordidxDict[eid])
    logger.info('dev set:{} non-rumor, {} false-rumor, {} true-rumor, {} un-rumor'.format(l1,l2,l3,l4))

    logger.info('loading test set')
    testPath = 'dataset/nfold/RNNtestSet_{}{}_tree.txt'.format(dataset, fold)
    sequences_test, index_test, y_test = [], [], []
    l1,l2,l3,l4 = 0,0,0,0
    for line in open(testPath):
        eid = line.strip()
        if eid not in labelDict: continue
        if eid not in sequencesDict: continue
        if len(sequencesDict[eid]) <= 0:
            continue
        label = labelDict[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        sequences_test.append(sequencesDict[eid])
        index_test.append(wordidxDict[eid])
    logger.info('test set:{} non-rumor, {} false-rumor, {} true-rumor, {} un-rumor'.format(l1,l2,l3,l4))

    logger.info('num of train set:{}, numn of dev set:{}, num of test set:{}'.format(len(sequences_train), len(sequences_dev), len(sequences_test)))

    y_train = torch.LongTensor(y_train)
    y_dev = torch.LongTensor(y_dev)
    y_test = torch.LongTensor(y_test)

    return sequences_train, index_train, y_train, sequences_dev, index_dev, y_dev, sequences_test, index_test, y_test

def evaluation_4class(prediction, y): # 4 dim
    prediction = prediction.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    TP4, FP4, FN4, TN4 = 0, 0, 0, 0
    e, RMSE, RMSE1, RMSE2, RMSE3, RMSE4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(y)):
        y_i, p_i = list(y[i]), list(prediction[i])
        ##RMSE
        for j in range(len(y_i)):
            RMSE += (y_i[j]-p_i[j])**2
        RMSE1 += (y_i[0]-p_i[0])**2
        RMSE2 += (y_i[1]-p_i[1])**2
        RMSE3 += (y_i[2]-p_i[2])**2
        RMSE4 += (y_i[3]-p_i[3])**2
        ## Pre, Recall, F
        Act = str(y_i.index(max(y_i))+1)
        Pre = str(p_i.index(max(p_i))+1)

        #print y_i, p_i
        #print Act, Pre
        ## for class 1
        if Act == '1' and Pre == '1': TP1 += 1
        if Act == '1' and Pre != '1': FN1 += 1
        if Act != '1' and Pre == '1': FP1 += 1
        if Act != '1' and Pre != '1': TN1 += 1
        ## for class 2
        if Act == '2' and Pre == '2': TP2 += 1
        if Act == '2' and Pre != '2': FN2 += 1
        if Act != '2' and Pre == '2': FP2 += 1
        if Act != '2' and Pre != '2': TN2 += 1
        ## for class 3
        if Act == '3' and Pre == '3': TP3 += 1
        if Act == '3' and Pre != '3': FN3 += 1
        if Act != '3' and Pre == '3': FP3 += 1
        if Act != '3' and Pre != '3': TN3 += 1
        ## for class 4
        if Act == '4' and Pre == '4': TP4 += 1
        if Act == '4' and Pre != '4': FN4 += 1
        if Act != '4' and Pre == '4': FP4 += 1
        if Act != '4' and Pre != '4': TN4 += 1
    ## print result
    Acc_all = round( float(TP1+TP2+TP3+TP4)/float(len(y)+e), 4 )
    Acc1 = round( float(TP1+TN1)/float(TP1+TN1+FN1+FP1+e), 4 )
    Prec1 = round( float(TP1)/float(TP1+FP1+e), 4 )
    Recll1 = round( float(TP1)/float(TP1+FN1+e), 4 )
    F1 = round( 2*Prec1*Recll1/(Prec1+Recll1+e), 4 )

    Acc2 = round( float(TP2+TN2)/float(TP2+TN2+FN2+FP2+e), 4 )
    Prec2 = round( float(TP2)/float(TP2+FP2+e), 4 )
    Recll2 = round( float(TP2)/float(TP2+FN2+e), 4 )
    F2 = round( 2*Prec2*Recll2/(Prec2+Recll2+e), 4 )

    Acc3 = round( float(TP3+TN3)/float(TP3+TN3+FN3+FP3+e), 4 )
    Prec3 = round( float(TP3)/float(TP3+FP3+e), 4 )
    Recll3 = round( float(TP3)/float(TP3+FN3+e), 4 )
    F3 = round( 2*Prec3*Recll3/(Prec3+Recll3+e), 4 )

    Acc4 = round( float(TP4+TN4)/float(TP4+TN4+FN4+FP4+e), 4 )
    Prec4 = round( float(TP4)/float(TP4+FP4+e), 4 )
    Recll4 = round( float(TP4)/float(TP4+FN4+e), 4 )
    F4 = round( 2*Prec4*Recll4/(Prec4+Recll4+e), 4 )

    microF = round( (F1+F2+F3+F4)/4,5 )
    RMSE_all = round( ( RMSE/len(y) )**0.5, 4)
    RMSE_all_1 = round( ( RMSE1/len(y) )**0.5, 4)
    RMSE_all_2 = round( ( RMSE2/len(y) )**0.5, 4)
    RMSE_all_3 = round( ( RMSE3/len(y) )**0.5, 4)
    RMSE_all_4 = round( ( RMSE4/len(y) )**0.5, 4)
    RMSE_all_avg = round( ( RMSE_all_1+RMSE_all_2+RMSE_all_3+RMSE_all_4 )/4, 4)
    # return ['acc:', Acc_all, 'Favg:',microF, RMSE_all, RMSE_all_avg,
    #         'C1:',Acc1, Prec1, Recll1, F1,
    #         'C2:',Acc2, Prec2, Recll2, F2,
    #         'C3:',Acc3, Prec3, Recll3, F3,
    #         'C4:',Acc4, Prec4, Recll4, F4]

    return Acc_all, microF, RMSE_all, RMSE_all_avg, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3,Recll3, F3, Acc4, Prec4, Recll4, F4

def convert_to_one_hot(y, C):
    # return np.eye(C)[y.reshape(-1)]
    return torch.zeros(y.shape[0], C).scatter_(1, y, 1)
