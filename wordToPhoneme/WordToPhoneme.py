#-*- coding: UTF-8 -*-
import cPickle as pickle

# 编码
def encoding(phonemeout):
    valueofcases = {'AA1': 0,
                    'AA2': 0,
                    'AE': 1,
                    'AH0': 2,
                    'AO': 3,
                    'AW': 4,
                    'AY': 5,
                    'AY1': 5,
                    'AY2': 5,
                    'B': 6,
                    'CH': 7,
                    'D': 8,
                    'DH': 9,
                    'EH': 10,
                    'EH1': 10,
                    'ER0': 11,
                    'EY': 12,
                    'F': 13,
                    'G': 14,
                    'HH': 15,
                    'IH1': 16,
                    'IH0': 16,
                    'IY0': 17,
                    'IY1': 17,
                    'JH': 18,
                    'K': 19,
                    'L': 20,
                    'M': 21,
                    'N': 22,
                    'NG': 23,
                    'OW0': 24,
                    'OY': 25,
                    'P': 26,
                    'R': 27,
                    'S': 28,
                    'SH': 29,
                    'T': 30,
                    'TH': 31,
                    'UH': 32,
                    'UW': 33,
                    'UW1': 33,
                    'V': 34,
                    'W': 35,
                    'Y': 36,
                    'Z': 37,
                    'ZH': 38
                    }
    switch = valueofcases.get(phonemeout, 'error')
    return switch


# 将从字典里读出的数存入数组
def splitstr(encodeout):
    container = []
    container = encodeout.split(' ')
    # print "container:", container
    return container


# 将数组里的元素编码
def cyclecode(splitout):
    b = []
    for i in splitout:
        b.append(encoding(i))
    print b


# load字典
def loaddict(inputword):
    f1 = file('/users/xuangemac/desktop/phonmedict.pkl', 'rb')
    temdict = pickle.load(f1)
    f1.close()
    print temdict[inputword]
    return temdict[inputword]


# 生成字典
def dumpdict(PathofRead,PathofDump):
    f1 = open(PathofRead, 'r')
    f2 = file(PathofDump, 'wb')
    phonemedict = {}
    for eachLine in f1:
        (container, dumped) = eachLine.split('\n', 1)
        (key, value) = container.split('  ', 1)
        phonemedict[key] = value
    pickle.dump(phonemedict, f2, True)
    f1.close()
    f2.close()

# 输入一个单词，可返回音位，注意输入时全部大写！
if __name__ == '__main__':
    a = raw_input()
    cyclecode(splitstr(loaddict(a)))
