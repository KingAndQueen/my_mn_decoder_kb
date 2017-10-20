
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data, my_load_task_movies, my_load_task_tt,Vocab
# from sklearn import cross_validation, metrics
from sklearn import model_selection, metrics
from memn2n import MemN2N
#from memn2n.my_seq2seq import sequence_loss, sequence_loss_by_example
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import pdb
import math
import nltk

tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 50, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 80.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 50, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")  # should consider the size of validation set
tf.flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 500, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 128, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 1, "Maximum size of memory.")
tf.flags.DEFINE_integer("additional_info_memory_size", 6, "size of additional info from KB . at least above 6")
tf.flags.DEFINE_integer("generate_rnn_layers",3, "the num layers of RNN.")
tf.flags.DEFINE_integer("generate_rnn_neurons", 128, "the number of neurons in one layer of RNN.")

tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
# tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("data_dir", "my_data/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "Directory to save checkpoints")
tf.flags.DEFINE_string("summary_path", "./summary/", "Directory to save summary")
tf.flags.DEFINE_string("model_type", "training", "whether to load the checkpoint sor training new model")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train = []
test = []
# train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
# train_sqa_movie,test_sqa_movie=my_load_task_movies(FLAGS.data_dir,20)
# train.extend(train_sqa_movie)
# test.extend(test_sqa_movie)
train_sqa_tt, test_sqa_tt = my_load_task_tt(FLAGS.data_dir + 'ticktock_data_small', 20)
train.extend(train_sqa_tt)
test.extend(test_sqa_tt)
# pdb.set_trace()
data = train + test
# train,test=model_selection.train_test_split(data,test_size=0.2,random_state=FLAGS.random_state)
words = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a  in data)))

#word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
#idx_word = {}

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
answer_size = max(map(len, (a for _, _, a in data)))
del data
sentence_size = max(query_size, sentence_size, answer_size)  # for the position
sentence_size += 1  # +1 for time words +1 for go +1 for eos

memory_size = min(FLAGS.memory_size, max_story_size) + FLAGS.additional_info_memory_size
vocab=Vocab()
vocab.add_vocab(words)
S, Q, A, A_fact, A_weight = vectorize_data(train, vocab, sentence_size, memory_size)
# Add time words/indexes
for i in range(memory_size):
    vocab.word_to_index('time{}'.format(i + 1))
additional_vocab_size = 50  # for additional infor from knowledge base
vocab_size = vocab.vocab_size+ additional_vocab_size  # +1 for nil word

# sentence_size= max(sentence_size,20) # set the same certain length for decoder
print('Vocabulary size:', vocab_size)
print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)
print('Memory size', memory_size)
# train/validation/test sets
# pdb.set_trace()

del train
trainS, valS, trainQ, valQ, trainA, valA, trainA_fact, valA_fact, trainA_weight, valA_weight = model_selection.train_test_split(
    S, Q, A, A_fact, A_weight, test_size=.2,
    random_state=FLAGS.random_state)  # validate set size have to equal to one batch size
del S, Q, A, A_fact, A_weight
testS, testQ, testA, testA_fact, testA_weight = vectorize_data(test, vocab, sentence_size, memory_size)
del test

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

# train_labels = np.argmax(trainA_fact, axis=1)
# train_sents = np.argmax(trainA, axis=2)
# test_labels = np.argmax(testA_fact, axis=1)
# test_sents = np.argmax(testA, axis=2)
# val_labels = np.argmax(valA_fact, axis=1)
# val_sents = np.argmax(valA, axis=2)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]


def show_idx2words(seq, vocab):
    #    idx_word={k:v for v,k in word_idx.iteritems()}
    words = []
    #    pdb.set_trace()
    if isinstance(seq, (list, np.ndarray)):
        for idx in seq:
            words.append(vocab.index_to_word(idx))
            if vocab.index_to_word(idx) == 'EOS':
                return words
        return words
    if isinstance(seq, (str, int)):
        return vocab.index_to_word(seq)


def count_bleu(labels_sents, predicts, vocab):
    #    pdb.set_trace()
    bleu = 0.0
    if len(labels_sents) != len(predicts):
        print('length is not same to count bleu')
        pdb.set_trace()
    for idx, sents in enumerate(predicts):
        preds_words = show_idx2words(sents, vocab)
        labels_words = show_idx2words(labels_sents[idx], vocab)
        #	bleu+=nltk.translate.bleu_score.sentence_bleu(labels_words,preds_words)
        bleu += nltk.translate.bleu([labels_words], preds_words)
    return bleu / len(predicts)


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0

def train_model(sess, model, vocab):
    print('Training...')
    train_summary_writer = tf.summary.FileWriter(FLAGS.summary_path, sess.graph)
    for t in range(1, FLAGS.epochs + 1):
        # Stepped learning rate
        #	print ('epoch:',t)
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            a_w = trainA_weight[start:end]
            cost_t = model.batch_fit(s, q, a,a_w, lr)
        if t % FLAGS.evaluation_interval == 0:
            # pdb.set_trace()

            train_pred_loss, train_pred_sents,summary = model.predict(s, q, a,a_w, pred_type='train')
            train_summary_writer.add_summary(summary,t)
            val_loss = 0.0
            sign=0
            val_pred_sents=[]
            for start in range(0, n_val, batch_size):
                sign += 1
                end = start + batch_size
                if end > n_val: continue
                s = valS[start:end]
                q = valQ[start:end]
                a = valA[start:end]
                a_w= valA_weight[start:end]
                val_pred_loss, val_pred_sent = model.predict(s, q, a,a_w, pred_type='valid')
                val_loss += val_pred_loss
                val_pred_sents+=list(val_pred_sent)
                # pdb.set_trace()

            print('-----------------------')
            print('Epoch', t)
            print('valid epoches number',sign)
            print('Training loss:',train_pred_loss)
            print('Validation loss:',val_loss/sign)
            # show the sentence generate quality
            #perplex_train = math.exp(float(train_pred_loss) if train_pred_loss < 500 else float('inf'))
           # print('Training sentence perplex:', perplex_train)
            #perplex_val = math.exp(float(val_loss/sign)) if val_loss/sign < 500 else float('inf')
            #print('Validation sentence perplex:', perplex_val)
            # pdb.set_trace()
            #train_preds_sents_words = np.argmax(train_pred_sents, 2)
            #            train_acc = metrics.accuracy_score(train_preds_sents_words[:,0], train_labels[:len(train_preds_sents_words)])
            #	    print('Training Accuracv:',train_acc)
           # bleu_score = count_bleu(a, train_preds_sents_words, vocab)
           # print('Training Bleu score:', bleu_score)

           # val_preds_sents_words = np.argmax(val_pred_sents, 2)
          #  bleu_score = count_bleu(valA[:len(val_preds_sents_words)], val_preds_sents_words, vocab)
          #  print('Validation Bleu score:', bleu_score)
            print('-----------------------')
    model.saver.save(sess, FLAGS.checkpoint_path, global_step=FLAGS.epochs)

    # test_model(model,testS,testQ,test_labels,batch_size)
    test_model(model, vocab)


def test_model(model, vocab):
    # def test_model(model,testS,testQ,test_labels,batch_size):

    test_preds_facts = []
    test_preds_sents = []
    loss=0
    for start in range(0, len(testS), batch_size):
        #     pdb.set_trace()
        end = start + batch_size
        if end > len(testS): continue  # throw away data less than a batch
        s = testS[start:end]
        q = testQ[start:end]
        a=testA[start:end]
        a_w=trainA_weight[start:end]
        test_result, word_idx_new = model.predict(s, q, a, a_w,pred_type='test')
        test_loss,test_pred_sents = test_result
      #  test_preds_facts += list(test_pred_facts)
        test_preds_sents += list(test_pred_sents)
    #    pdb.set_trace()
      #  loss_batch = sequence_loss(test_pred_sents, test_sents[start:end],
                            #  testA_weight[start:end])
        loss+=test_loss
    loss=loss/(len(testS)/batch_size)
    print ('Test loss:',loss)
   # perplex_test=math.exp(float(loss)) if loss<500 else float('inf')
    #print('Test sentance perplex:', perplex_test)

    #    test_preds = model.predict(testS, testQ,pred_type='test')
    #  test_acc = metrics.accuracy_score(test_preds_facts, test_labels[:len(test_preds_facts)])
    #  print ('labels:\n',show_idx2words(test_labels,word_idx))
    #  print("MN Testing Accuracy:", test_acc)
   # test_preds_sents_words = np.argmax(test_preds_sents, 2)
   # bleu_score = count_bleu(testA[:len(test_preds_sents_words)], test_preds_sents_words, vocab)
    #print('Test Bleu score:', bleu_score)
    #   test_acc = metrics.accuracy_score(test_preds_sents_words[:,0], test_labels[:len(test_preds_sents_words)])
    #   print("Testing Accuracy:", test_acc)
   # test_preds_sents = np.argmax(test_preds_sents, 2)


    # for index in range(len(test_preds_sents_words)):
    #     print ('test question:',show_idx2words(testQ[index],vocab))
    #    # print ('test pred facts:',show_idx2words(test_preds_facts[index],idx_word))
    #     print ('test pred sents:',show_idx2words(test_preds_sents_words[index],vocab))
    #     print ('test label:',show_idx2words(testA[index],vocab))
    #     print ('\n')

with tf.Session() as sess:
    print('Initial model...')
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, vocab=vocab,
                   additional_info_size=FLAGS.additional_info_memory_size,
                   num_layers=FLAGS.generate_rnn_layers, rnn_size=FLAGS.generate_rnn_neurons)

    if FLAGS.model_type == 'training':
        print('Initial model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
        train_model(sess, model, vocab)
    else:
        print('Reading model parameters from checkpoints %s', FLAGS.checkpoint_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_model(model, vocab)
