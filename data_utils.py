from __future__ import absolute_import

import os
import re
import numpy as np
import pdb
import copy
import json
import nltk

class Vocab():
    def __init__(self,word2idx=None):
        self.word2idx={'<eos>':0,'<go>':1,'<pad>':2,'<unk>':3}
        self.idx2word={0:'<eos>',1:'<go>',2:'<pad>',3:'<unk>'}
    def add_vocab(self,words):
        if isinstance(words, (list, np.ndarray)):
            for word in words:
                if word not in self.word2idx:
                    index=len(self.word2idx)
                    self.word2idx[word]=index
                    self.idx2word[index]=word
        else:
            if words not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[words] = index
                self.idx2word[index] = words
    def word_to_index(self,word):
        self.add_vocab(word)
        return self.word2idx[word]
    def index_to_word(self,index):
        if index in self.idx2word:
            return self.idx2word[index]
        else:
            return '<unk>'
    @property
    def vocab_size(self):
        return len(self.idx2word)

def my_get_friends(data_dir, data_type, filter_sents_len):
    data_base=[]
    data_dir = data_dir + '/' + data_type
    f=open(data_dir,'r')
    # f = open(data_dir, 'r',encoding = 'utf-8')
    last_sents=[]
    s,q,a=[],[],[]
    for lines in f:
        # lines = lines.strip()[2:-5]
        # pdb.set_trace()
        lines=lines.split()[:filter_sents_len]
        temp=' '
        lines=temp.join(lines)
        if len(lines)>2:
            sents=lines[lines.index(':')+1:]
            if len(last_sents)==0:
                last_sents=sents
                continue
            q=tokenize(last_sents.strip())
            a=tokenize(sents.strip())
            s.append(q)
            last_sents=sents
        else:
            data_base.append((s,q,a))
            s=[]
            q=[]
            a=[]
            last_sents=[]
    # pdb.set_trace()
    return data_base

def my_load_friends(data_dir,fileter_sents_len):
    train_data=my_get_friends(data_dir,'train.txt',fileter_sents_len)
    test_data=my_get_friends(data_dir,'test.txt',fileter_sents_len)
    valid_data = my_get_friends(data_dir, 'validation.txt', fileter_sents_len)
    train_data+=valid_data
    return train_data, test_data
def my_load_task_tt(data_dir, filter_sents_len):
    train_data = my_get_tt(data_dir, 'train', filter_sents_len)
    test_data = my_get_tt(data_dir, 'test', filter_sents_len)
    #    database = load_json.LoadData([data_dir])
    #  pdb.set_trace()
    return train_data, test_data

#    with open('word2vec/test_complete.pkl','w') as f:
#	pickle.dump(database,f)
def my_get_tt(data_dir, data_type, filter_sents_len):
    filelist = list()
    data_dir = data_dir + '/' + data_type
    if not os.path.exists(data_dir):
        print ('data_dir is not exist!')
        return None
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            file_name = os.path.splitext(os.path.join(root, name))
            if file_name[1] == '.json':
                filelist.append(os.path.join(root, name))
    return my_load_data_tt(filelist, filter_sents_len)


def my_load_data_tt(filelist, filter_sents_len):
    database = []
    for datafile in filelist:
        f = open(datafile)
        line = f.readline()
        f.close()
        raw_data = json.loads(str(line.strip()))
        for data in raw_data:
            s = []
            q = nltk.word_tokenize(data['question'])
            a = nltk.word_tokenize(data['answer'])
            s.append(q)
            if len(q) < filter_sents_len and len(a) < filter_sents_len:
                database.append((s, q, a))

    return database


def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    #   pdb.set_trace()
    return train_data, test_data


def my_load_task_movies(path, filter_sents_len):
    ''' get the train and test of movie's subtitles
    '''
    train_data = my_get_movies(path + 'movies_task1_qa_train.txt', filter_sents_len)
    test_data = my_get_movies(path + 'movies_task1_qa_test.txt', filter_sents_len)
    return train_data, test_data


def my_get_movies(path, filter_sents_len):
    with open(path) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            story = []
            nid, line = line.split(' ', 1)
            if '\t' in line:
                q, a = line.split('\t')
                q = tokenize(q)
                a = tokenize(a)
                if len(q) >= filter_sents_len or len(a) >= filter_sents_len: continue
                story.append(q)
                data.append((story, q, a))
            #    pdb.set_trace()
    return data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return sent.split()  # [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    #    pdb.set_trace()
    for line in lines:
        #  line = str.lower(line) #I do not understand why the author lower words
        nid, line = line.split(' ', 1)
        #	print nid #close vim then run or will error
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = tokenize(a)
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, vocab, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    vocab_size=vocab.vocab_size
    S = []
    Q = []
    A = []
    A_fact = []
    A_weight = []
    for story, query, answer in data:
        ss = []
        # word in sentences to index code
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([vocab.word_to_index(w) for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            #   ss[i][-1] = vocab_size - memory_size - i + len(ss) # its affection is to sequence the facts
            ss[i][-1] = vocab_size - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [vocab.word_to_index(w) for w in query] +  [vocab.word_to_index('<pad>')] * lq

        y = [vocab.word_to_index('<go>')]
        weight = [1.0]
        for answer_ in answer:
            y.append(vocab.word_to_index(answer_))
            weight.append(1.0)
        A_fact.append(copy.copy(y[0]))  # use the first word as fact in BAbI task
        y.append(vocab.word_to_index('<eos>'))
        weight.append(1.0)
        la = max(0, sentence_size+1 - len(y))
        if la < 0: pdb.set_trace()
        for temp in range(la):
            y.append(vocab.word_to_index('<pad>'))
            weight.append(0.0)
        #	pdb.set_trace()
        #	if len(y)!=sentence_size
        if len(weight)>sentence_size:weight.pop()
        S.append(ss)
        Q.append(q)
        # A.append(np.array(y,'float64'))
        A.append(y)
        A_weight.append(np.array(weight))

    return np.array(S), np.array(Q), np.array(A), np.array(A_fact), np.array(A_weight)


#if __name__ == '__main__':
   # load_ticktock_data()
