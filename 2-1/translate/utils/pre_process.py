import unicodedata
import re
import numpy as np
import torch



# 定义一个语言类，方便进行自动的建立、词频的统计等
# 在这个对象中，最重要的是两个字典：word2index，index2word
# 故名思议，第一个字典是将word映射到索引，第二个是将索引映射到word
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # 在语言中添加一个新句子，句子是用空格隔开的一组单词
        # 将单词切分出来，并分别进行处理
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # 插入一个单词，如果单词已经在字典中，则更新字典中对应单词的频率
        # 同时建立反向索引，可以从单词编号找到单词
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
# 将unicode编码转变为ascii编码
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 把输入的英文字符串转成小写
def normalizeEngString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 对输入的单词对做过滤，保证每句话的单词数不能超过MAX_LENGTH
def filterPair(p, MAX_LENGTH_PATCH, MAX_LENGTH_DESCRIPTION):
    return len(p[0].split(' ')) < MAX_LENGTH_PATCH and \
        len(p[1].split(' ')) < MAX_LENGTH_DESCRIPTION

# 输入一个句子，输出一个单词对应的编码序列
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# 和上面的函数功能类似，不同在于输出的序列等长＝MAX_LENGTH
def indexFromSentence(lang, EOS_token, sentence, MAX_LENGTH):
    indexes = indexesFromSentence(lang, sentence)
    for i in range(MAX_LENGTH - len(indexes)):
        indexes.append(EOS_token)
    if len(indexes)>MAX_LENGTH:
        print(len(indexes))


    return(indexes)

# 从一个词对到下标
def indexFromPair(input_lang, output_lang, EOS_token, MAX_LENGTH_PATCH, MAX_LENGTH_DESCRIPTION, pair):
    input_variable = indexFromSentence(input_lang, EOS_token, pair[0], MAX_LENGTH_PATCH)
    target_variable = indexFromSentence(output_lang, EOS_token, pair[1], MAX_LENGTH_DESCRIPTION)
    return (input_variable, target_variable)

# translate a index list to a sentence
def SentenceFromList(lang, lst, EOS_token):
    result = [lang.index2word[i] for i in lst if i != EOS_token]
    if lang.name == 'French':
        result = ' '.join(result)
    else:
        result = ' '.join(result)
    return(result)

# 计算准确度的函数
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    # print("@@@", pred, labels)
    rights = pred.eq(labels.data).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素

