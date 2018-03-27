# encoding=utf-8

__author__ = 'Jingzhang'

# add ner features

import argparse
import numpy as np
import random
import time
from datetime import date
import os

from gensim.models.keyedvectors import KeyedVectors
from time import gmtime, strftime
from tensor_board_logger import BoardLogger

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
import logging

torch.cuda.set_device(6)
CUDA = True   # use gpu
FEATURE = True
# Constants from C++ code
WORD_DIM = 300
EMBEDDING_DIM = 300 + 45 + 1 if  FEATURE else 300
#+ 45 + 1  # cat word embedding | pos_tag_embedding | first_letter_capital_embedding
LAYERS = 1
INPUT_DIM = 300 # + 45 + 1 if FEATURE else 300
XCRIBE_DIM = 128
SEG_DIM = 150
H1DIM = 128
H2DIM = 128
TAG_DIM = 5
DURATION_DIM = 6
POS_TAG_DIM = 45

FEATURE_DIM = POS_TAG_DIM + 1 if FEATURE else 0

# lstm builder: LAYERS, XCRIBE_DIM, SEG_DIM, m?
# (layers, input_dim, hidden_dim, model)

DATA_MAX_SEG_LEN = 6

MAX_SENTENCE_LEN = 32
MINIBATCH_SIZE = 1
BATCH_SIZE = 5000

use_dropout = False
dropout_rate = 0.0
ner_tagging = False
use_max_sentence_len_training = False

LABELS = ['LOC', 'PER', 'ORG', 'MISC', '0']
I_LABELS = ['0', 'I']
TAGS = ['LOC', 'PER', 'ORG', 'MISC']
POS_TAG = ['PRP$', 'VBG', 'VBD', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', \
            'WP', 'VBZ', 'DT', '"', 'RP', '$', 'NN', ')', '(', 'FW', 'POS', \
            '.', 'TO', 'PRP', 'RB', ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'LS', \
            'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$', 'NN|SYM', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'UH']

board_logger = None

def FloatTensor(x):
    return torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
def Variable(x):
    return autograd.Variable(x).cuda() if CUDA else autograd.Variable(x)

def logsumexp(inputs, dim=None, keepdim=False):
        return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)


########## SegRNN model ###############
class SegRNN(nn.Module):
    def __init__(self):
        super(SegRNN, self).__init__()
        self.forward_context_initial = (nn.Parameter(0.02*torch.randn(1, 1, XCRIBE_DIM)), nn.Parameter(0.02*torch.randn(1, 1, XCRIBE_DIM)))
        self.backward_context_initial = (nn.Parameter(0.02*torch.randn(1, 1, XCRIBE_DIM)), nn.Parameter(0.02*torch.randn(1, 1, XCRIBE_DIM)))
        self.forward_context_lstm = nn.LSTM(INPUT_DIM, XCRIBE_DIM)
        self.backward_context_lstm = nn.LSTM(INPUT_DIM, XCRIBE_DIM)
        self.register_parameter("forward_context_initial_0", self.forward_context_initial[0])
        self.register_parameter("forward_context_initial_1", self.forward_context_initial[1])
        self.register_parameter("backward_context_initial_0", self.backward_context_initial[0])
        self.register_parameter("backward_context_initial_1", self.backward_context_initial[1])

        self.forward_initial = (nn.Parameter(0.02*torch.randn(1, 1, SEG_DIM)), nn.Parameter(0.02*torch.randn(1, 1, SEG_DIM)))
        self.backward_initial = (nn.Parameter(0.02*torch.randn(1, 1, SEG_DIM)), nn.Parameter(0.02*torch.randn(1, 1, SEG_DIM)))
        self.Y_encoding = nn.Parameter(torch.eye(TAG_DIM, TAG_DIM)).split(1, 0)

        self.Z_encoding = [nn.Parameter(torch.ones(1, 1, DURATION_DIM)) for i in range(1, DATA_MAX_SEG_LEN + 1)]

        self.register_parameter("forward_initial_0", self.forward_initial[0])
        self.register_parameter("forward_initial_1", self.forward_initial[1])
        self.register_parameter("backward_initial_0", self.backward_initial[0])
        self.register_parameter("backward_initial_1", self.backward_initial[1])
        #for idx, encoding in enumerate(self.Y_encoding):
        #    self.register_parameter("Y_encoding_" + str(idx), encoding)
        for idx, encoding in enumerate(self.Z_encoding):
            self.register_parameter("Z_encoding_" + str(idx), encoding)

        self.forward_lstm = nn.LSTM(2 * XCRIBE_DIM, SEG_DIM)
        self.backward_lstm = nn.LSTM(2 * XCRIBE_DIM, SEG_DIM)
        self.V = nn.Linear(2 * SEG_DIM  + DURATION_DIM + FEATURE_DIM, TAG_DIM) # srnn only predict segmentation, not tag

        self.W = nn.Linear(TAG_DIM, 1)
        self.Phi = nn.Tanh()
        self.type_linear = nn.Linear(2 * XCRIBE_DIM + WORD_DIM + FEATURE_DIM, TAG_DIM)
        self.type_softmax = nn.LogSoftmax()
        self.type_criterion = nn.NLLLoss()

    def calc_loss(self, batch_data, batch_label):
        N, B, K = batch_data.shape
        # print(B, len(batch_label))
        # print(N, B, K)
        batch_data = FloatTensor(batch_data)

        sentence_data = batch_data[:, :, 0:WORD_DIM]
        #print sentence_data.shape
        forward_xcribe_data, backward_xcribe_data, forward_precalc, backward_precalc = self._precalc(sentence_data)

        log_alphas = [Variable(torch.zeros((1, B, 1)))]
        for i in range(1, N + 1):
            t_sum = []
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(1, B, 1)
                # print ([self.Y_encoding[y] for y in range(len(LABELS))])
                #y_encoding_expand = torch.cat([self.Y_encoding[y] for y in range(len(LABELS))], 0).repeat(1, B, 1)
                z_encoding_expand = self.Z_encoding[i - j - 1].repeat(1, B, 1)

                if CUDA:
                    #y_encoding_expand = y_encoding_expand.cuda()
                    z_encoding_expand = z_encoding_expand.cuda()

                #print 'y_encoding_expand shape', y_encoding_expand.data.shape
                #print 'z_encoding_expand shape', z_encoding_expand.data.shape
                #print 'precalc_expand shape', precalc_expand.data.shape
                if FEATURE:
                    feature_data = torch.cat([FloatTensor((i - j) * torch.mean(batch_data[j:i, :, WORD_DIM:-1], 0)),\
                                              batch_data[j, :, -1].repeat(1, 1)], 1)
                    '''
                    for w in range(0, B):
                        for x in range(j + 1, i):
                            for y in range(len(batch_data[x,w])):
                                if batch_data[x, w][y] is 1:
                                    feature_data[0, w][y] = 1
                    '''
                    feature_encoding = Variable(feature_data).repeat(1, B, 1)

                    # LABELS, MINIBATCH, FEATURES
                    seg_encoding = torch.cat([precalc_expand, z_encoding_expand,feature_encoding\
                                          ], 2)
                else:
                    seg_encoding = torch.cat([precalc_expand, z_encoding_expand], 2)

                # Linear thru features: LABELS, MINIBATCH, 1
                #print 'seg_encoding', seg_encoding
                t = self.W(self.Phi(self.V(seg_encoding )))
                #print 't:', t
                # summed across labels: 1, MINIBATCH, 1
                summed_t = logsumexp(t, 0, True)
                t_sum.append(log_alphas[j] + summed_t)
            # cat across seglenths: SEG_LENGTH, MINIBATCH, 1
            all_t_sums = torch.cat(t_sum, 0)
            # sum across lengths: 1, MINIBATCH, 1
            new_log_alpha = logsumexp(all_t_sums, 0, True)
            log_alphas.append(new_log_alpha)

        loss = torch.sum(log_alphas[N])
        type_output_encoding = []
        type_gold_encoding = []

        for batch_idx in range(B):
            indiv = Variable(torch.zeros(1))
            chars = 0
            label = batch_label[batch_idx]
            # feature_encoding = batch_data[:,batch_idx, WORD_DIM:]
            for tag, length in label:
                # print (tag, length)
                if length >= DATA_MAX_SEG_LEN:
                    continue
                forward_val = forward_precalc[chars][chars + length - 1][:, batch_idx, np.newaxis, :]
                backward_val = backward_precalc[chars][chars + length - 1][:, batch_idx, np.newaxis, :]
                #y_val = self.Y_encoding[0]
                z_val = self.Z_encoding[length - 1]

                #print 'length:', length
                #print('forward_val:', forward_val.data.shape)
                #print('backward_val:', backward_val.data.shape)
                #print('y_val:', y_val.data.shape)
                #print('z_val:', z_val.data.shape)


                if FEATURE:
                    feature_val = Variable(torch.cat([length * torch.mean(batch_data[chars:chars+length, batch_idx, WORD_DIM:-1], 0).repeat(1, 1, 1),  \
                                                      FloatTensor([batch_data[chars, batch_idx, -1]]).repeat(1, B, 1)], 2))

                    '''
                    for x in range(chars+1, chars + length):
                        for y in range(len(batch_data[x, batch_idx])):
                            if batch_data[x, batch_idx][y] is 1:
                                if feature_val[0, batch_idx][y] == 0:
                                    feature_val[0, batch_idx][y] = 1
                                    print ("add 1 to feature vec")
                                    print('feature_val:', feature_val.data)
                    '''

                    seg_encoding = torch.cat([forward_val, backward_val,  z_val, feature_val\
                                          ], 2)
                else:
                    seg_encoding = torch.cat([forward_val, backward_val, z_val], 2)
                #print 'z_val,', z_val
                #print 'feature_val', feature_val
                #print 'seg_encoding',seg_encoding
                #print self.W(self.Phi(self.V(seg_encoding)))
                indiv += self.W(self.Phi(self.V(seg_encoding)))[0][0]

                if CUDA:
                    forward_xcribe_chars = forward_xcribe_data[chars].cuda()
                    backward_xcribe_chars = backward_xcribe_data[chars + length - 1].cuda()
                    # type loss part
                else:
                    forward_xcribe_chars = forward_xcribe_data[chars]
                    backward_xcribe_chars = backward_xcribe_data[chars + length - 1]
                #print('forward_xcribe_data:', forward_xcribe_chars.type)
                #print('backward_xcribe_data', backward_xcribe_chars.type)
                #print('sentence data:', sentence_data[chars:chars+length:,:].type)
                #print('average mention:', self._average(sentence_data[chars:chars+length]).repeat(1, B, 1).type)
                #print('feature val:', feature_val.type)
                #print ('chars:', chars, 'chars + length - 1:', chars + length - 1)
                mention_avg = self._average(sentence_data[chars:chars+length, :, :]).repeat(1, B, 1)
                if FEATURE:
                    type_input_encoding = torch.cat([forward_xcribe_chars, backward_xcribe_chars, \
                                                     mention_avg, feature_val], 2)
                else:
                    type_input_encoding = torch.cat([forward_xcribe_chars, backward_xcribe_chars,\
                                                     mention_avg], 2)

                #logging.info('segment and mention cosine similarity:', F.cosine_similarity(torch.cat([forward_val, backward_val]), mention_avg))
                #print('type_input_coding:', type_input_encoding)
                output = self.type_linear(type_input_encoding)
                #print('output:', output[0].type)
                type_output = self.type_softmax(output[0])
                #print('training type output:', type_output)
                type_output_encoding.append(type_output)

                #print('Y_encoding:', self.Y_encoding)
                #tmp_tensor = torch.FloatTensor(1, 1, TAG_DIM).zero_()
                type_gold = LABELS.index(tag)
                type_gold_encoding.append(type_gold)
                #print(type_gold)
                #type_loss = type_loss + nn.NLLLoss(type_output_encoding, gold_type_encoding)

                chars += length
            loss -= indiv
        #type_loss_sum = sum(type_loss)
        #print('type loss:', type_loss_sum)
        # type loss part
        type_output_encoding = torch.cat(type_output_encoding)
        type_gold_encoding = Variable(torch.LongTensor(type_gold_encoding))
        type_loss = self.type_criterion(type_output_encoding, type_gold_encoding)

        #print('output_encoding: ' , type_output_encoding)
        #print('gold encoding:', type_gold_encoding)
        #print('type loss:', type_loss)
        return loss, type_loss

    def _average(self, data):
        return Variable(torch.mean(torch.cat(data, 0), 0))

    def _precalc(self, data):
        N, B, K = data.shape

        forward_xcribe_data = []
        hidden = (
            torch.cat([self.forward_context_initial[0] for b in range(B)], 1),
            torch.cat([self.forward_context_initial[1] for b in range(B)], 1)
        )
        for i in range(N):
            next_input = Variable(data[i, :])
            out, hidden = self.forward_context_lstm(next_input.view(1, B, K), hidden)
            forward_xcribe_data.append(out)

        backward_xcribe_data = []
        hidden = (
            torch.cat([self.backward_context_initial[0] for b in range(B)], 1),
            torch.cat([self.backward_context_initial[1] for b in range(B)], 1)
        )
        for i in range(N - 1, -1, -1):
            next_input = Variable(data[i, :])
            out, hidden = self.backward_context_lstm(next_input.view(1, B, K), hidden)
            backward_xcribe_data.append(out)


        xcribe_data = []
        for i in range(N):
            xcribe_data.append(torch.cat([forward_xcribe_data[i], backward_xcribe_data[i]], 2))

        forward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # forward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = (
                torch.cat([self.forward_initial[0] for b in range(B)], 1),
                torch.cat([self.forward_initial[1] for b in range(B)], 1)
            )
            for j in range(i, min(N, i + DATA_MAX_SEG_LEN)):
                next_input = xcribe_data[j]
                out, hidden = self.forward_lstm(next_input, hidden)
                forward_precalc[i][j] = out


        backward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # backward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = (
                torch.cat([self.backward_initial[0] for b in range(B)], 1),
                torch.cat([self.backward_initial[1] for b in range(B)], 1)
            )
            for j in range(i, max(-1, i - DATA_MAX_SEG_LEN), -1):
                next_input = xcribe_data[j]
                out, hidden = self.backward_lstm(next_input, hidden)
                backward_precalc[j][i] = out

        return forward_xcribe_data, backward_xcribe_data, forward_precalc, backward_precalc

    def infer(self, data):
        data = FloatTensor(data)
        N, B, K = data.shape    # N: sentence length, B: batch_size, K: word_dim
        if N == 0:
            return []
        #print("infer data.shape:", data.shape)
        sentence_data = data[:, :, 0:WORD_DIM]
        #print 'sentence data shape', sentence_data.shape
        forward_xcribe_data, backward_xcribe_data, forward_precalc, backward_precalc = self._precalc(sentence_data)
        #print 'forward_precalc size', len(forward_precalc), len(forward_precalc[0])
        #print 'backward_precalc size', len(backward_precalc), len(backward_precalc[0])

        log_alphas = [('0', -1, 0.0)]
        for i in range(1, N + 1):
            t_sum = []
            max_len = -1
            max_t = float("-inf")
            max_label = -1
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(1, B, 1)

                z_encoding_expand = self.Z_encoding[i - j - 1].repeat(1, B, 1)
                #feature_data = torch.FloatTensor(data[i - 1, :, WORD_DIM:])
                if FEATURE:
                    feature_data = torch.cat([FloatTensor((i-j) * torch.mean(data[j:i, :, WORD_DIM:-1], 0)), \
                                              data[j, :, -1].repeat(1, 1)], 1)
                    if CUDA:
                        #y_encoding_expand = y_encoding_expand.cuda()
                        z_encoding_expand = z_encoding_expand.cuda()
                    '''
                    for w in range(0, B):
                        for x in range(j + 1, i):
                            for y in range(len(data[x,w])):
                                if data[x, w][y] is 1:
                                    feature_data[0, w][y] = 1
                    '''
                    feature_encoding = Variable(feature_data).repeat(1, B, 1)


                    seg_encoding = torch.cat([precalc_expand, z_encoding_expand, feature_encoding\
                                      ], 2)
                else:
                    seg_encoding = torch.cat([precalc_expand, z_encoding_expand,], 2)
                t_val = self.W(self.Phi(self.V(seg_encoding)))
                #print 'function inference, t_val:', t_val
                t = t_val + log_alphas[j][2]
                '''
                # print("t_val: ", t_val)
                for y in range(len(LABELS)):
                    if t.data[y, 0, 0] > max_t:
                        max_t = t.data[y, 0, 0]
                        max_label = y
                        max_len = i - j
                        '''
                if t.data[0, 0, 0] > max_t:
                    max_t = t.data[0, 0, 0]
                    max_len = i - j

            #print('max_len:', max_len)
            #print('i:', i)
            #print('forward_xcribe_data:', forward_xcribe_data[i - max_len])
            #print('backward_xcribe_data[i-1]', backward_xcribe_data[i-1])
            #print('mention average:', self._average(sentence_data[i - max_len: i, :, :]).repeat(1, B, 1))
            precalc_data = torch.cat([forward_xcribe_data[i - max_len], backward_xcribe_data[i-1], self._average(sentence_data[i - max_len:i, :, :]).repeat(1, B, 1)], 2).repeat(1, B, 1)

            z_encoding_expand = self.Z_encoding[max_len - 1].repeat(1, B, 1)
                #feature_data = torch.FloatTensor(data[i - 1, :, WORD_DIM:])
            if FEATURE:
                feature_data = FloatTensor(max_len * torch.mean(data[i-max_len:i, :, WORD_DIM:], 0))
                if CUDA:
                    #y_encoding_expand = y_encoding_expand.cuda()
                    z_encoding_expand = z_encoding_expand.cuda()
                '''
                for w in range(0, B):
                    for x in range(i - max_len + 1, i):
                        for y in range(len(data[x,w])):
                            if data[x, w][y] is 1:
                                feature_data[0, w][y] = 1
                '''
                feature_encoding = Variable(feature_data).repeat(1, B, 1)
                #print('feature_encoding:', feature_encoding)
                predict_type = self.type_softmax(self.type_linear(torch.cat([precalc_data, feature_encoding], 2)))
            else:
                predict_output = self.type_linear(precalc_data)
                #print('predict_output', predict_output)
                predict_type = self.type_softmax(predict_output[0])
                #print('predict_type', predict_type)
            top_value, top_index = torch.topk(predict_type, 1)
            #print('predict_type', predict_type)
            #print 'top index', top_index
            predict_tag = LABELS[int(top_index[0][0])]
            #print('predict_tag:', predict_tag, 'predicted len', max_len)
            log_alphas.append((predict_tag, max_len, max_t))

        cur_pos = N
        ret = []
        while cur_pos != 0:
            ret.append((log_alphas[cur_pos][0], log_alphas[cur_pos][1]))
            cur_pos -= log_alphas[cur_pos][1]
        ret = list(reversed(ret))
        #logging.info('infer result:' + str(ret))
        return ret

def parse_embedding(embed_filename):
    embed_file = open(embed_filename)
    embedding = dict()
    for line in embed_file:
        values = line.split()
        embedding[values[0]] = np.array(values[1:]).astype(np.float)
    return embedding


def parse_file(train_filename, embedding):
    train_file = open(train_filename)
    sentences = []
    labels = []

    for line in train_file:
        line = line.strip().split("|||")
        sentence = line[0].split(" ")

        sentence.pop()
        # print len(sentence), sentence
        label_list = line[1].split(" ")

        # fill sentence embedding

        N = len(sentence)

        if 'train' in train_filename and N == 0:
            pass
        else:

            sentence_vec = np.zeros((N, EMBEDDING_DIM))
            for i in range(0, N):
                split_list = sentence[i].split('&&')
                if len(split_list) < 3: # avoid the situation of the mark '-DOCSTART'
                    pass
                    #print sentence[i]
                else:
                    c = split_list[0].lower()

                    pos_tag = split_list[1]
                    if '&' in pos_tag:
                        pos_tag = pos_tag.replace('&','')
                    capital = split_list[2]
                    pos_tag_embed = [0 for j in range(len(POS_TAG))]
                    pos_tag_embed[POS_TAG.index(pos_tag)] = 1
                    capital_embed = [ord(capital)-ord('0')]

                    #print len(list(embedding[c]))
                    #print pos_tag_embed
                    if c in embedding:
                        if FEATURE:
                            sentence_vec[i, :] =np.concatenate((list(embedding[c]), pos_tag_embed, capital_embed), 0)
                        else:
                            sentence_vec[i, :] = embedding[c]
                    else:
                        if FEATURE:
                            sentence_vec[i, :] = np.concatenate((list(embedding["unknown"]), pos_tag_embed, capital_embed))
                        else:
                            sentence_vec[i, :] = embedding["unknown"]

            sentences.append(sentence_vec)
            #print 'vec len:', len(sentence_vec)
            #print pos_tag_embed, capital_embed

            # fill label
            label_sum = 0
            label = []
            for part in label_list:
                part = part.split(";")

                try:
                    entity_url = part[0]
                    tag = part[1]
                    duration = int(part[2])
                except:
                    print "invalid detected. line:", sentence, "part:", part

                if (label_sum + duration) <= N:
                    label_sum += duration
                label.append((tag, duration))
            labels.append(label)
    return sentences, labels


def parse_file_content(filename):
    reader = open(filename)
    sentences = []
    for line in reader:
        line = line.strip().split("|||")
        sentence = line[0].lower()
        sentences.append(sentence)
    return sentences


# count correct labels with tag
def count_correct_labels(predicted, gold, with_tag=True):


    if with_tag:
        index = 0
        correct_count = 0.0
        predicted_set = set()
        chars = 0
        gold_len = 0

        for (tag, l) in predicted:
            label = (tag, chars, chars + l)
            #print 'tag:', tag
            if tag in TAGS:
                predicted_set.add(label)
            chars += l
        chars = 0
        # print "predicted set:", predicted_set
        # print "gold labels:"
        for (tag, l) in gold:
            label = (tag, chars, chars + l)
            # print label,
            if tag in TAGS:
                gold_len += 1
                if label in predicted_set:
                    correct_count += 1
            chars += l
            index += 1
        # print
        # print 'correct cuont:', correct_count
        return correct_count, len(predicted_set), gold_len
    else:
        correct_count = 0.0
        predicted_set = set()
        chars = 0.0
        gold_len = 0.0

        for (tag, l) in predicted:
            if tag in TAGS:
                tag = 'I'
            label = (tag, chars, chars + l)
            if tag is 'I':
                predicted_set.add(label)
            chars += l
        chars = 0
        index = 0
        for (tag, l) in gold:
            if tag in TAGS:
                tag = 'I'
            label = (tag, chars, chars + l)
            if tag is 'I':
                gold_len += 1.0
                if label in predicted_set:
                    correct_count += 1
            chars += l
            index += 1
        return correct_count, len(predicted_set), gold_len


def eval_f1(seg_rnn, pairs, contents, testfilename, with_tag):
    t = strftime("%Y-%m-%d %H:%M:%S")

    if with_tag:
        writer = open(testfilename + t + "_with_tag.out",'w')
    else:
        writer = open(testfilename + t + "_without_tag.out",'w')
    gold_segs = 0.0
    predicted_segs = 0.0
    correct_segs = 0.0
    doc_correct = 0.0
    doc_predicted = 0.0
    doc_gold = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    doc_sum = 0
    for idx, (datum, gold_labels) in enumerate(pairs):

        # '-DOCSTART-' is the separator of documents, a batch is a sentence
        # calculate document level correctness
        if cmp(contents[idx], '-docstart-') == 0:
            writer.write('********-DOCSTART-***********\n')

            #print 'doc start', doc_sum

            if doc_predicted == 0.0:
                doc_precision = 0.0
            else:
                doc_precision = float(doc_correct)/doc_predicted

            if doc_gold == 0.0:
                doc_recall = 0.0
            else:
                doc_recall = float(doc_correct)/doc_gold
                doc_sum += 1

            precision_sum += doc_precision
            recall_sum += doc_recall

            writer.write('doc_precision:' + str(doc_precision) + ', doc_recall:' + str(doc_recall) \
                         + ', doc_correct:' + str(doc_correct) + ', doc_predict:' + str(doc_predicted) + ', doc_gold:' + str(doc_gold) + '\n')

            doc_correct = 0.0
            doc_predicted = 0.0
            doc_gold = 0.0
        else:
            # get sentence correctness
            predicted_label = seg_rnn.infer(datum.reshape(len(pairs[idx][0]), 1, EMBEDDING_DIM))
            writer.write('data shape:' + str(len(datum)) + "\n")
            writer.write('sentence length:' + str(len(pairs[idx][0])) + "\n")
            writer.write('predict:' + str(predicted_label) + "\n")
            writer.write('gold:' + str(gold_labels) + "\n")

            correct_seg, predicted_seg, gold_seg = count_correct_labels(predicted_label, gold_labels, with_tag)
            doc_correct += correct_seg
            doc_predicted += predicted_seg
            doc_gold += gold_seg
            if correct_seg < predicted_seg or correct_seg < gold_seg:
                writer.write('error, ' + 'index:' + str(idx) + ', content:' + contents[idx] + '\n')

        correct_segs += correct_seg
        writer.write('correct_segs:' + str(correct_segs) + '\n')
    writer.write('doc_sum:' + str(doc_sum))
    if predicted_segs > 0:
        writer.write("Micro Precision: " + str(float(correct_segs) / predicted_segs))
    if gold_segs > 0:
        writer.write(", Micro Recall: " + str(float(correct_segs) / gold_segs) + "\n")

    prec = precision_sum/doc_sum
    rec = recall_sum/doc_sum
    if prec + rec == 0.0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec + rec)
    writer.write('Macro Precision:' + str(prec))
    writer.write(', Macro Recall:' + str(rec))
    writer.write(', F1:' + str(f1))
    return f1, prec, rec


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmental RNN.')
    parser.add_argument('--train', help='Training file')
    parser.add_argument('--test', help='Test file')
    parser.add_argument('--embed', help='Character embedding file')
    parser.add_argument('--model', help='Saved model')
    parser.add_argument('--lr', help='Learning rate (default=0.01)')
    parser.add_argument('--with_tag', help='evaluate ner with or without tag')
    parser.add_argument('--model_path', help='path the srnn model to be saved')
    parser.add_argument('--CUDA', help='1 to use gpu, 0 to use cpu')
    parser.add_argument('--log_dir', help='the directory to store log files')
    args = parser.parse_args()

    if args.with_tag is not None:
        if cmp(args.with_tag, '0') == 0:
            args.with_tag = False
        else:
            args.with_tag = True
        print 'with_tag:', args.with_tag
    else:
        args.with_tag = True

    if args.embed is not None:
        if cmp(args.embed.split('.')[-1], 'txt') == 0:
            embedding = parse_embedding(args.embed)
        else:
            embedding = KeyedVectors.load_word2vec_format(args.embed, binary=True)
    else:
        embedding = parse_embedding("./data/embed/glove.6B.300d.txt")
    print("Done parsing embedding")

    if args.train is not None:
        data, labels = parse_file(args.train, embedding)
        pairs = list(zip(data, labels))
        print("Done parsing training data")

    if args.test is not None:
        test_data, test_labels = parse_file(args.test, embedding)
        test_pairs = list(zip(test_data, test_labels))
        contents = parse_file_content(args.test)            # get sentence text

        print("Done parsing testing data")
    if args.log_dir is None:
        args.log_dir = '/data/zj/log/'

    if args.CUDA is not None:
        if args.CUDA is '1':
            CUDA = True
        else:
            CUDA = False

    if args.model is not None:
        seg_rnn = torch.load(args.model)
    else:
        seg_rnn = SegRNN()
    if CUDA:
        seg_rnn.cuda()

    random.seed(1338)
    if args.lr is not None:
        learning_rate = float(args.lr)
    else:
        learning_rate = 0.001
    optimizer = torch.optim.SGD(seg_rnn.parameters(), lr=learning_rate)

    ########### log config ###############
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    rootlogger = logging.getLogger()
    t = strftime("%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(os.path.join(args.log_dir, 'logging', t + '.log'))
    logFormat = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler.setFormatter(logFormat)
    rootlogger.addHandler(fileHandler)
    logging.info('The now time is ' +  t)
    board_path = os.path.join(args.log_dir, 'train', args.model_path.split('/')[-1] + t)
    os.makedirs(board_path, mode=0o777)
    board_logger = BoardLogger(board_path)
    #######################################
    args.model_path = args.model_path + t

    count = 0.0
    sum_loss = 0.0
    correct_count = 0.0
    sum_gold = 0.0
    sum_pred = 0.0
    if args.train is not None:

        max_f1 = 0.0
        max_prec = 0.0
        max_rec = 0.0
        max_iter = 0
        for batch_num in range(10):
            random.shuffle(pairs)
            start_time = time.time()
            for i in range(0, min(BATCH_SIZE, len(pairs)), MINIBATCH_SIZE):

                # seg_rnn.train()
                optimizer.zero_grad()

                iter_count = batch_num * min(BATCH_SIZE, len(pairs)) + i

                if use_max_sentence_len_training:
                    max_len = MAX_SENTENCE_LEN
                    batch_size = min(MINIBATCH_SIZE, len(pairs) - i)
                else:
                    max_len = len(pairs[i][0])
                    batch_size = 1
                batch_data = np.zeros((max_len, batch_size, EMBEDDING_DIM))
                # print batch_data.shape
                batch_labels = []
                for idx, (datum, label) in enumerate(pairs[i:i+batch_size]):
                    batch_data[:, idx, :] = datum
                    batch_labels.append(label)
                    # print datum, label
                seg_loss, type_loss = seg_rnn.calc_loss(batch_data, batch_labels)
                #print('type loss in main:', type_loss)

                sum_loss += (seg_loss.data[0] + type_loss.data[0])/len(batch_data)
                count += 1.0 * batch_size
                loss = seg_loss + type_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm(seg_rnn.parameters(), 5.0)
                if i % 50 == 0:
                    board_logger.add_scalar_summary('loss', loss/len(batch_data), iter_count)
                    board_logger.add_scalar_summary('seg loss', seg_loss.data[0]/len(batch_data), iter_count)
                    board_logger.add_scalar_summary('type loss', type_loss.data[0]/len(batch_data), iter_count)
                optimizer.step()
                if i % 50 == 0:
                    for name, p in seg_rnn.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            name = name.replace('.', '/')
                            board_logger.add_histo_summary('value/'+ name, p, iter_count)
                            board_logger.add_histo_summary('grad/' + name, p.grad, iter_count)
                if i % 100 == 0:
                    logging.info("Batch "+str(batch_num) + ", datapoint "
                                + str(i) + ", avg loss " + str(sum_loss / count))

                sentence_len = len(pairs[i][0])
                # print 'pred data:', batch_data[0:sentence_len, 0, np.newaxis, :]
                pred = seg_rnn.infer(batch_data[0:sentence_len, 0, np.newaxis, :])
                gold = pairs[i][1]
                # print("prediction:", pred)
                # print("gold:", gold)
                # print(pairs[i][1])
                correct_count_tmp, predict_tmp, gold_tmp = count_correct_labels(pred, gold, args.with_tag)
                correct_count += correct_count_tmp
                sum_pred += predict_tmp
                sum_gold += gold_tmp


                cum_prec = correct_count / sum_pred if sum_pred != 0 else 0.0
                cum_rec = correct_count / sum_gold if sum_gold != 0 else 0.0

                if i % 100 == 0:
                    prec_tmp = correct_count_tmp/predict_tmp if predict_tmp > 0 else 0.0
                    rec_tmp = correct_count_tmp/gold_tmp if gold_tmp > 0 else 1.0
                    board_logger.add_scalar_summary('precision', cum_prec, iter_count)
                    board_logger.add_scalar_summary('recall', cum_rec, iter_count)
                    if cum_prec > 0 and cum_rec > 0:
                        board_logger.add_scalar_summary('f1', 2.0*cum_prec*cum_rec/(cum_prec+cum_rec), iter_count)
                    sum_gold = 0.0
                    sum_pred = 0.0
                    correct_count = 0.0


                if i % 1000 == 0 and cum_prec > 0 and cum_rec > 0:
                    cum_f1 = 2.0 / (1.0 / cum_prec + 1.0/cum_rec)
                    logging.info("F1:" + str(cum_f1) +
                                  ", precision:" + str(cum_prec)  + ", recall:" + str(cum_rec))
                # print(seg_rnn.Y_encoding[0], seg_rnn.Y_encoding[5])
                # print(seg_rnn.Y_encoding[0].grad, seg_rnn.Y_encoding[5].grad)
                #for param in seg_rnn.parameters():
                #    print(param)

                if i % 1000 == 0:
                    end_time = time.time()
                    time_took = end_time - start_time
                    logging.info("Took" + str(time_took) + "to run 1000 training sentences")
                    start_time = time.time()
                    # eval_f1(seg_rnn, test_pairs, contents, args.test, args.with_tag)


            if args.test is not None:
                f1, prec, rec = eval_f1(seg_rnn, test_pairs, contents, args.test, args.with_tag)
                logging.info('eval f1:' + str(f1) + ', precision:' + str(prec) + ', recall:' +str( rec))

                if f1 > max_f1:
                    max_f1 = f1
                    max_prec = prec
                    max_rec = rec
                    max_iter = batch_num
            torch.save(seg_rnn, args.model_path + "iter_" + str(batch_num) + '.model')

    logging.info('max f1:' + str(max_f1) + ', prec:' + str(max_prec) + ', max_rec:'+str(max_rec) +
                 ', max_iter: ' + str(max_iter))

    if args.model_path is not None:
        torch.save(seg_rnn, args.model_path)
    if args.test is not None:
        #torch.save(seg_rnn, args.model_path)
        eval_f1(seg_rnn, test_pairs, contents, args.test, args.with_tag)
