from io import open
from pickle import TRUE
from pyexpat.errors import codes
import unicodedata
import string
import re
import random
import argparse

import constrants as Const

#Pytorch必备的包
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as DataSet
criterion = nn.NLLLoss()

from model.models import *
from utils.pre_process import *
import numpy as np

#from model.evaluate import evaluation

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_model(params, codelabelsize, descriptionlabelsize, n_layers):
    """
        Use args to initialize the appropriate model
    """
    # select encoder
    if params.encodermodel == "e_GRU":
        encoder = EncoderRNN(codelabelsize, params.hidden_size, n_layers = n_layers)
    
    # select decoder
    if params.decodermodel == "d_GRU":
        decoder = DecoderRNN(params.hidden_size, descriptionlabelsize, n_layers = n_layers)
    
    # use gpu or not
    # if use_cuda:
    #     if params.gpu:
    #         encoder = encoder.to(device)
    #         decoder = decoder.to(device)
    return encoder, decoder


def train(params, train_loader, valid_loader, encoder, decoder):
    encoder.train()
    decoder.train()
    
    if use_cuda:
        #encoder = nn.DataParallel(encoder)
        encoder = encoder.to(device)
        #decoder = nn.DataParallel(decoder)
        decoder = decoder.to(device)
        
    
    # Define optimizers for both encoder and decoder 
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=params.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=params.lr)

    plot_losses = []
    ratio_max = 0
    for epoch in range(params.num_epoch):
        print_loss_total = 0
        # loop in training dataet
        for data in train_loader:
            # input_variable = Variable(data[0]).cuda() if use_cuda else Variable(data[0])
            input_variable = Variable(data[0]).to(device) if use_cuda else Variable(data[0])
            # size of input_variable：(batch_size, length_seq), eg: (32, 100)
            target_variable = Variable(data[1]).to(device) if use_cuda else Variable(data[1])
            # size of target_variable：(batch_size, length_seq), eg: (32, 100)
            encoder_hidden = encoder.initHidden(data[0].size()[0])
            # data[0].size()->torch.Size([batch_size, seq_length]), 
            # size of encoder_hidden: (number_layer, batch_size, hidden_size), (eg: 2, 32, 16)
            # clean gradient of encoder and decoder
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # Beginning of Encoder
            encoder_outputs, encoder_hidden = encoder(input_variable.to(device), encoder_hidden.to(device))
            # print("encoder_outputs", encoder_outputs.shape, encoder_outputs[0][49], encoder_outputs[0][0], encoder_hidden[0][0], encoder_hidden[1][0])
            # size of encoder_outputs：batch_size, length_seq, hidden_size*direction, eg: 32, 100, 16*2(cat)
            # size of encoder_hidden：direction*n_layer, batch_size, hidden_size; eg: (2, 32, 16) only take the last encoder_hidden from encoder outputs
            
            # Beginning of Decoder
            # Feed SOS_token sysbol to Decoder as the beginning

            
            decoder_input = Variable(torch.LongTensor([[SOS_token]] * target_variable.size()[0])) # torch.Size([32, 1])
            
            # decoder_input大小：(batch_size, 1)
            decoder_input = decoder_input.to(device) if use_cuda else decoder_input

            # 让解码器的隐藏层状态等于编码器的隐藏层状态
            decoder_hidden = encoder_hidden
            # size of decoder_hidden：direction*n_layer, batch_size, hidden_size; eg: (2, 32, 16)

            # 以teacher_forcing_ratio的比例用target中的翻译结果作为监督信息
            use_teacher_forcing = True if random.random() < params.teacher_forcing_ratio else False
            base = torch.zeros(target_variable.size()[0])
            if use_teacher_forcing:
                # 教师监督: 将下一个时间步的监督信息输入给解码器
                # 对时间步循环
                for di in range(MAX_LENGTH_DESCRIPTION):
                    # 开始一步解码
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    # decoder_ouput大小：batch_size, hidden_size
                    # 计算损失函数
                    loss += criterion(decoder_output, target_variable[:, di]) # size of target_variable: batch_size, 1
                    # print("output", decoder_output[0], "loss", criterion(decoder_output[0], target_variable[0, di]), target_variable[0, di])
                    # 将训练数据当做下一时间步的输入
                    decoder_input = target_variable[:, di].unsqueeze(1)  # Teacher forcing
                    # decoder_input大小：batch_size, length_seq
                    
            else:
                # 没有教师训练: 使用解码器自己的预测作为下一时间步的输入
                # 开始对时间步进行循环
                for di in range(MAX_LENGTH_DESCRIPTION):
                    # 进行一步解码
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    
                    #decoder_ouput大小：batch_size, output_size(vocab_size)
                    
                    #从输出结果（概率的对数值）中选择出一个数值最大的单词作为输出放到了topi中
                    topv, topi = decoder_output.data.topk(1, dim = 1) # k=1, topv is value, topi means index

                    
                    # if we want use random top3, we can define value here
                    
                    #topi 尺寸：batch_size, k
                    ni = topi[:, 0] # (batch_size)                    
                    #print("topi", topi, ni)

                    # 将输出结果ni包裹成Variable作为解码器的输入
                    decoder_input = Variable(ni.unsqueeze(1)) # (batch_size, 1)
                    # decoder_input大小：batch_size, 1
                    # decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                    decoder_input = decoder_input.to(device) if use_cuda else decoder_input

                    #计算损失函数
                    loss += criterion(decoder_output, target_variable[:, di])
                    # print("begin", decoder_output, target_variable[:, di], decoder_output.shape, target_variable[:, di].shape)
            
                
            
            # backward
            loss.backward()
            loss = loss.cpu() if use_cuda else loss
            # 开始梯度下降
            encoder_optimizer.step()
            decoder_optimizer.step()
            # 累加总误差
            print_loss_total += loss.data.numpy()

        # 计算训练时候的平均误差
        print_loss_avg = print_loss_total / len(train_loader)

        torch.save(encoder, "%s/coregenencoder.mdl"% Const.PARAMETERS_DIR)
        torch.save(decoder, "%s/coregendecoder.mdl"% Const.PARAMETERS_DIR)
            
        # 开始跑校验数据集
        valid_loss = 0
        rights = []
        
        # 对校验数据集循环
        for data in valid_loader:
            
            # input_variable = Variable(data[0]).cuda() if use_cuda else Variable(data[0])
            # # input_variable的大小：batch_size, length_seq
            # target_variable = Variable(data[1]).cuda() if use_cuda else Variable(data[1])
            input_variable = Variable(data[0]).to(device) if use_cuda else Variable(data[0]) 
            # size of input_variable：(batch_size, length_seq), eg: 32, 100
            target_variable = Variable(data[1]).to(device) if use_cuda else Variable(data[1])
            # target_variable的大小：batch_size, length_seq

            encoder_hidden = encoder.initHidden(data[0].size()[0])

            loss = 0
            encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
            # encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
            # encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

            decoder_input = Variable(torch.LongTensor([[SOS_token]] * target_variable.size()[0]))
            # decoder_input大小：batch_size, length_seq
            # decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_input = decoder_input.to(device) if use_cuda else decoder_input

            decoder_hidden = encoder_hidden
            # decoder_hidden大小：direction*n_layer, batch_size, hidden_size

            # 没有教师监督: 使用解码器自己的预测作为下一时间步解码器的输入
            for di in range(MAX_LENGTH_DESCRIPTION):
                # 一步解码器运算
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                #decoder_ouput大小：batch_size, output_size(vocab_size)
                
                # 选择输出最大的项作为解码器的预测答案
                topv, topi = decoder_output.data.topk(1, dim = 1)
                #topi 尺寸：batch_size, k
                ni = topi[:, 0]
                decoder_input = Variable(ni.unsqueeze(1))
                # decoder_input大小：batch_size, length_seq
                # decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                decoder_input = decoder_input.to(device) if use_cuda else decoder_input
                
                # 计算预测的准确率，记录在right中，right为一个二元组，分别存储猜对的个数和总数
                right = rightness(decoder_output, target_variable[:, di])
                rights.append(right)
                
                # 计算损失函数
                loss += criterion(decoder_output, target_variable[:, di])
            loss = loss.cpu() if use_cuda else loss
            # 累加校验时期的损失函数
            valid_loss += loss.data.numpy()
        # 打印每一个Epoch的输出结果
        right_ratio = 1.0 * torch.sum(torch.Tensor([i[0] for i in rights])) / torch.sum(torch.Tensor([i[1] for i in rights]))

        if right_ratio> ratio_max:
            ratio_max=right_ratio
            torch.save(encoder, "%s/coregen_best_encoder.mdl"% Const.PARAMETERS_DIR)
            torch.save(decoder, "%s/coregen_best_decoder.mdl"% Const.PARAMETERS_DIR)
        print('process：%d%% train loss：%.4f，valid loss：%.4f，word accuracy：%.2f%%' % (epoch * 1.0 / params.num_epoch * 100, 
                                                        print_loss_avg,
                                                        valid_loss / len(valid_loader),
                                                        100.0 * right_ratio))
        # 记录基本统计指标
        plot_losses.append([print_loss_avg, valid_loss / len(valid_loader), right_ratio])

def evaluation(test_X, test_Y, encoder, decoder, input_lang, EOS_token):
    encoder = encoder.eval()
    decoder = decoder.eval()

    preds = []
    ground_truths = []

    # 对每个句子进行循环
    for ind in range(len(test_X)):
        data = [test_X[ind]]
        target = [test_Y[ind]]
        # 把源语言的句子打印出来
        #print(SentenceFromList(input_lang, data[0], EOS_token))
        input_variable = Variable(torch.LongTensor(data)).to(device) if use_cuda else Variable(torch.LongTensor(data))
        # input_variable的大小：batch_size, length_seq
        target_variable = Variable(torch.LongTensor(target)).to(device) if use_cuda else Variable(torch.LongTensor(target))

        # target_variable的大小：batch_size, length_seq

        # 初始化编码器
        encoder_hidden = encoder.initHidden(input_variable.size()[0])

        loss = 0
        
        # 编码器开始编码，结果存储到了encoder_hidden中
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
        # encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
        # encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

        # 将SOS作为解码器的第一个输入
        decoder_input = Variable(torch.LongTensor([[SOS_token]] * target_variable.size()[0]))
        # decoder_input大小：batch_size, length_seq
        decoder_input = decoder_input.to(device) if use_cuda else decoder_input

        # 将编码器的隐含层单元数值拷贝给解码器的隐含层单元
        decoder_hidden = encoder_hidden
        # decoder_hidden大小：direction*n_layer, batch_size, hidden_size

        # 没有教师指导下的预测: 使用解码器自己的预测作为解码器下一时刻的输入
        output_sentence = []
        # decoder_attentions = torch.zeros(max_length, max_length)
        # decoder_attentions = torch.zeros(MAX_LENGTH_DESCRIPTION, MAX_LENGTH_DESCRIPTION)
        rights = []
        # 按照输出字符进行时间步循环
        for di in range(MAX_LENGTH_DESCRIPTION):
            # 解码器一个时间步的计算
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            #decoder_ouput大小：batch_size, output_size(vocab_size)
            
            # 解码器的输出
            topv, topi = decoder_output.data.topk(1, dim = 1)
            #topi 尺寸：batch_size, k
            ni = topi[:, 0]
            decoder_input = Variable(ni.unsqueeze(1))
            # print("decoder_input", decoder_input)
            ni = ni.cpu().numpy()[0] # 切换到cpu模式
            # ni = ni.numpy()[0]
            # print(ni, EOS_token, type(ni.item()), type(EOS_token))

            if ni.item() is EOS_token:
                break

            
            
            # 将本时间步输出的单词编码加到output_sentence里面
            output_sentence.append(ni)
            # decoder_input大小：batch_size, length_seq
            decoder_input = decoder_input.to(device) if use_cuda else decoder_input
            
            # 计算输出字符的准确度

            
            right = rightness(decoder_output, target_variable[:, di])
            # print("##", decoder_output, target_variable[:, di], di, right)
            rights.append(right)
        # 解析出编码器给出的翻译结果
        sentence = SentenceFromList(output_lang, output_sentence, EOS_token)
        # 解析出标准答案
        standard = SentenceFromList(output_lang, target[0], EOS_token)
        
        # 将句子打印出来
        print('pred result：', sentence)
        print('ground truth：', standard)
        preds.append(sentence)
        ground_truths.append(standard)
        
        # 输出本句话的准确率
        # right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        # print('word accuracy_score：', 100.0 * right_ratio)
        # print('\n')
    if len(preds) == len(ground_truths):
        with open('result/preds.txt', 'w') as f:
            for pd in preds:
                f.write(str(pd)+'\n')
        with open('result/ground_truths.txt', 'w') as f:
            for gt in ground_truths:
                f.write(str(gt)+'\n')

def read_args():
    parser = argparse.ArgumentParser(description="train a representation network for patch")
    # general setting of architecture
    parser.add_argument("encodermodel", type=str, choices=["e_GRU", "e_conv_attn", "e_patchformer"], help="encodermodel")
    parser.add_argument("decodermodel", type=str, choices=["d_GRU", "d_Transformer"], help="decodermodel")
    parser.add_argument("-num_epoch", type=int, help="number of epochs to train")
    parser.add_argument("-batch_size", type=int, required = True, default=16, help="batch size")
    parser.add_argument("-lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    # Training our model
    parser.add_argument('-train', type = int, help='training Patch2vec model')
    # parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    # parser.add_argument('-dictionary_data', type=str, default='./data/lmg/dict.pkl', help='the directory of our dicitonary data')
    parser.add_argument('-teacher_forcing_ratio', type= float, default= 0.5, help= ' use value of teacher_forcing_ratio to supervise the results of translation')
    parser.add_argument('-hidden_size', type = int, default =16, help='hidden_size')
    parser.add_argument("-gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    return parser   

def cut(_str, max_len):
    if len(_str.split(' '))>max_len:
        _str = ' '.join(_str.split(' ')[:max_len-1])
    return _str

    
if __name__ == '__main__':
    params = read_args().parse_args()
    # 读取平行语料库
    # 英＝法
    lines = open('data/cleaned.train.diff', encoding = 'utf-8')
    train_patch = lines.read().strip().split('\n')
    lines = open('data/cleaned.train.msg', encoding = 'utf-8')
    train_description = lines.read().strip().split('\n')

    lines = open('data/cleaned.valid.diff', encoding = 'utf-8')
    valid_patch = lines.read().strip().split('\n')
    lines = open('data/cleaned.valid.msg', encoding = 'utf-8')
    valid_description = lines.read().strip().split('\n')

    lines = open('data/cleaned.test.diff', encoding = 'utf-8')
    test_patch = lines.read().strip().split('\n')
    lines = open('data/cleaned.test.msg', encoding = 'utf-8')
    test_description = lines.read().strip().split('\n')
    


    # 定义两个特殊符号，分别对应句子头和句子尾
    SOS_token = 0
    EOS_token = 1

    #对英文做标准化处理
    train_pairs = [[normalizeEngString(pat), normalizeEngString(des)] for pat, des in zip(train_patch, train_description)] # turn all words into lower case letters
    valid_pairs = [[normalizeEngString(pat), normalizeEngString(des)] for pat, des in zip(valid_patch, valid_description)] 
    test_pairs = [[normalizeEngString(pat), normalizeEngString(des)] for pat, des in zip(test_patch, test_description)] 

        
    # 处理数据形成训练数据
    # 设置句子的最大长度
    # MAX_LENGTH = 100
    MAX_LENGTH_PATCH = 256
    MAX_LENGTH_DESCRIPTION = 50   

    train_pairs = [[cut(pair[0], MAX_LENGTH_PATCH), cut(pair[1], MAX_LENGTH_DESCRIPTION)] for pair in train_pairs] 
    valid_pairs = [[cut(pair[0], MAX_LENGTH_PATCH), cut(pair[1], MAX_LENGTH_DESCRIPTION)] for pair in valid_pairs] 
    test_pairs = [[cut(pair[0], MAX_LENGTH_PATCH), cut(pair[1], MAX_LENGTH_DESCRIPTION)] for pair in test_pairs] 

    
    
    

    print("pairs", train_pairs[0])
    print('meaningful sentence pairs：', len(train_pairs)+len(valid_pairs)+len(test_pairs)) #有效句子对

    # 对句子对做过滤，处理掉那些超过MAX_LENGTH长度的句子
    input_lang = Lang('Patch')
    output_lang = Lang('Description')

    # build vocabularies for both patch and description
    for pair in train_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    for pair in valid_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    for pair in test_pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("total number of words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

  
     
    # 形成训练集，首先，打乱所有句子的顺序
    #random_idx = np.random.permutation(range(len(train_pairs)))
    #train_pairs = [train_pairs[i] for i in random_idx]

    

    # 将语言转变为单词的编码构成的序列
    train_pairs = [indexFromPair(input_lang, output_lang, EOS_token, MAX_LENGTH_PATCH, MAX_LENGTH_DESCRIPTION, pair) for pair in train_pairs]
    valid_pairs = [indexFromPair(input_lang, output_lang, EOS_token, MAX_LENGTH_PATCH, MAX_LENGTH_DESCRIPTION, pair) for pair in valid_pairs]
    test_pairs = [indexFromPair(input_lang, output_lang, EOS_token, MAX_LENGTH_PATCH, MAX_LENGTH_DESCRIPTION, pair) for pair in test_pairs]
    

    pairs = train_pairs

    batch_size = params.batch_size

    print('train length：', len(pairs))
    print('valid length：', len(valid_pairs))
    print('test length：', len(test_pairs))

    # 形成训练对列表，用于喂给train_dataset
    pairs_X = [pair[0] for pair in pairs]
    pairs_Y = [pair[1] for pair in pairs]
    valid_X = [pair[0] for pair in valid_pairs]
    valid_Y = [pair[1] for pair in valid_pairs]
    test_X = [pair[0] for pair in test_pairs]
    test_Y = [pair[1] for pair in test_pairs]

    
    # train dataset and dataloader
    print(len(pairs_X), len(pairs_X[0]),len(pairs_Y), len(pairs_Y[0]))

    train_dataset = DataSet.TensorDataset(torch.LongTensor(pairs_X), torch.LongTensor(pairs_Y))
    train_loader = DataSet.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=8)
    # valid dataset and dataloader
    valid_dataset = DataSet.TensorDataset(torch.LongTensor(valid_X), torch.LongTensor(valid_Y))
    valid_loader = DataSet.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers=8)
    # test dataset and dataloader
    test_dataset = DataSet.TensorDataset(torch.LongTensor(test_X), torch.LongTensor(test_Y))
    test_loader = DataSet.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)

    
    if params.train == 0:
        # 开始训练过程
        # 定义网络结构
        hidden_size = params.hidden_size
        n_layers = 1
        print("training !!!!")
        # encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers = n_layers)
        # # ipdb.set_trace()
        # decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers = n_layers)
        
        encoder, decoder = pick_model(params, input_lang.n_words, output_lang.n_words, n_layers)

        # decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.5, max_length = 5, n_layers = n_layers)
        # encoder = torch.load('%s/coregenencoder.mdl'% Const.PARAMETERS_DIR)
        # decoder = torch.load('%s/coregendecoder.mdl'% Const.PARAMETERS_DIR)
        train(params, train_loader, valid_loader, encoder, decoder)
        torch.save(encoder, '%s/coregenencoder.mdl'% Const.PARAMETERS_DIR)
        torch.save(decoder, '%s/coregendecoder.mdl'% Const.PARAMETERS_DIR)
    elif params.train == 1:
        # 开始evaluate过程
        # 定义网络结构
        hidden_size = params.hidden_size
        n_layers = 1
        print("training !!!!")
        print("input_lang.n_words", input_lang.n_words)
        # encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers = n_layers)
        # # input_lang.n_words 就是vocab_size
        # # ipdb.set_trace()
        # decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers = n_layers)

        encoder, decoder = pick_model(params, input_lang.n_words, output_lang.n_words, n_layers)

        encoder = torch.load('%s/coregenencoder.mdl'% Const.PARAMETERS_DIR)
        decoder = torch.load('%s/coregendecoder.mdl'% Const.PARAMETERS_DIR)

        # 首先，在测试集中随机选择20个句子作为测试
        # np.random.seed(2022)
        # indices = np.random.choice(range(len(test_X)), 20)



        encoder.eval()
        decoder.eval()
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        evaluation(test_X, test_Y, encoder, decoder, input_lang, EOS_token)
        