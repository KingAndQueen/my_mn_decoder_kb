"""
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
# import model.my_graph as my_graph
import pdb
import re

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

#from model.my_seq2seq import sequence_loss, sequence_loss_by_example


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2]..
    """
    #    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
    with tf.name_scope("add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def get_continuous_chunks(query):
    chunked = ne_chunk(pos_tag(word_tokenize(query)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    #    pdb.set_trace()
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        # elif current_chunk:  #i think it need to be modified like this
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        if len(i) > 1:
            if i[1] == 'CD':
                continuous_chunk.append(i[0])
                #            else:
                #                     continue
                #    pdb.set_trace()


                # following five lines are for data filter
                #    continuous_chunk=[]
                #    words=word_tokenize(query)
                #    for word in words:
                #	if not word.islower() and  word.isalpha():
                #	    continuous_chunk.append(word)

    return continuous_chunk


class Model_Mix(object):
    """End-To-End Memory Network."""

    def __init__(self, FLAGS,
                 encoding=position_encoding,
                 session=tf.InteractiveSession(),
                 vocab=None, ):
        """Creates an mix Network
        """
        self._sess = session
        self.process_type = FLAGS.process_type
        self._batch_size = FLAGS.batch_size
        self._vocab_size = FLAGS.vocab_size
        self._sentence_size = FLAGS.sentence_size
        self._memory_size = FLAGS.memory_size
        self._embedding_size = FLAGS.embedding_size
        self._hops = FLAGS.hops
        self._max_grad_norm = FLAGS.max_grad_norm
        self._name = FLAGS.model_type
        self._rnn_neurons = FLAGS.rnn_neurons
        self._num_layers = FLAGS.rnn_layers
        self._build_inputs(FLAGS)
        self._build_vars(FLAGS)
        self._position_encoder = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")
        self.vocab = vocab
        self.additional_info_size = FLAGS.additional_info_memory_size
        self._lr=tf.Variable(float(FLAGS.learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = tf.assign(self._lr,self._lr * FLAGS.learning_rate_decay_factor)
        self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        # logits_mem = None
        if FLAGS.model_type == 'memn2n' or FLAGS.model_type == 'mix':
            logits, logits_mem = self._inference(self._stories, self._queries)  # (batch_size, vocab_size)
            # count the logits of inference
            fact_labels = self._ans_fact
            cross_entropy_facts = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.cast(fact_labels, tf.float32),
                                                                          name='fact_cross_entropy')
            cross_entropy_facts_sum = tf.reduce_sum(cross_entropy_facts, name='fact_cross_entropy_sum')
            predict_key_fact = tf.argmax(logits, 1, name='predict_key_fact')
            self.predict_key_fact = predict_key_fact
            loss_op = cross_entropy_facts_sum

        if FLAGS.model_type == 'seq2seq' or FLAGS.model_type == 'mix':
            # decoder my docoder add a rnn in memorynetwork
            queries = tf.unstack(self._queries, axis=1)
            q_emb = [embedding_ops.embedding_lookup(self.rnn_embedding, query) for query in queries]
            answers = tf.unstack(self._answers, axis=1)
            a_emb = [embedding_ops.embedding_lookup(self.rnn_embedding, ans) for ans in answers]
            a_emb = a_emb[:self._sentence_size]
            if FLAGS.model_type == 'seq2seq': logits_mem = None
            encoder_state, attention_state = self.rnn_encoder(q_emb)
            rnn_outputs = self.rnn_decoder(encoder_state, attention_state, a_emb, logits_MemKG=logits_mem)
            # cross entropy
            predict_proba_op = tf.nn.softmax(rnn_outputs, name="predict_proba_op")
            self.predict_op = predict_proba_op

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_outputs,
                                                                           labels=self._answers_shifted,
                                                                           name="cross_entropy")
            cross_entropy_weighted = tf.multiply(cross_entropy, self._weight)
            cross_entropy_weighted_sum = math_ops.reduce_sum(cross_entropy_weighted)
            weight_sum = tf.reduce_sum(self._weight, axis=1)
            loss_op = cross_entropy_weighted_sum/weight_sum
            # pdb.set_trace()
        if FLAGS.model_type == 'mix':
            loss_op = cross_entropy_facts_sum + cross_entropy_weighted_sum

        loss_op=tf.reduce_mean(loss_op, name="cross_entropy_sum")
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]

        train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")
        # pdb.set_trace()
        tf.summary.scalar("loss", loss_op)
        # assign ops
        self.loss_op=loss_op
        self.loss_summary = tf.summary.merge_all()
        self.train_op = train_op
        self.saver = tf.train.Saver(tf.global_variables())
        # load machine code to English and English to Machine code map from data file
        # self._fb_map = my_graph.get_code_to_english('./useful_Map.txt')

    def rnn_encoder(self, q_emb):
        with tf.variable_scope('rnn_encoder'):
            single_cell = tf.nn.rnn_cell.GRUCell(self._rnn_neurons)
            cell = single_cell
            #    pdb.set_trace()
            if self._num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(self._num_layers)])
            encoder_output, encoder_state = rnn.static_rnn(cell, q_emb, dtype=tf.float32)
            top_output = [array_ops.reshape(o, [-1, 1, cell.output_size]) for o in encoder_output]
            attention_states = array_ops.concat(top_output, 1)

        return encoder_state, attention_states

    def rnn_decoder(self, encoder_state, attention_states, answers, logits_MemKG=None):

        #	if (q_emb[0].get_shape()[1]-logits_MemKG.get_shape()[1])>0:
        # padding the logits to have same length of the input of decoder rnn
        #	    logits_MemKG=tf.pad(logits_MemKG,[[0,0],[0,int(q_emb[0].get_shape()[1]-logits_MemKG.get_shape()[1])]])
        #	for i,query in enumerate(q_emb):
        #	    q_emb[i]=tf.add(logits_MemKG, q_emb[i]) # I should test more integrate methods
        #	    q_emb[i]=tf.concat(1,[logits_MemKG,q_emb[i]])
        #	pdb.set_trace()
        #	if initial_state==None:
        #		batch_size=q_emb[0].get_shape().with_rank_at_least(2)[0]
        #		initial_state=cell.zero_state(batch_size,tf.float32),
        with tf.variable_scope('rnn_decoder_cover'):
            num_heads = 1
            batch_size = answers[0].get_shape()[0]
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value
            hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            attention_vec_size = attn_size
            for a in range(num_heads):
                k = tf.get_variable('AttnW_%d' % a, [1, 1, attn_size, attention_vec_size])
                hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], 'SAME'))
                v.append(tf.get_variable('AttnV_%d' % a, [attention_vec_size]))

            def attention(query):
                ds = []
                if nest.is_sequence(query):
                    query_list = nest.flatten(query)
                    for q in query_list:
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = array_ops.concat(query_list, 1)
                for a in range(num_heads):
                    with tf.variable_scope('Attention_%d' % a):
                        y = linear(query, attention_vec_size, True)
                        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                        s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                        a = nn_ops.softmax(s)
                        d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                        ds.append(array_ops.reshape(d, [-1, attn_size]))
                return ds

            def extract_argmax_and_embed(prev, _):
                """Loop_function that extracts the symbol from prev and embeds it."""
                prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
                return embedding_ops.embedding_lookup(self.rnn_embedding, prev_symbol)

            if self.process_type == 'test':
                loop_function = extract_argmax_and_embed
            else:
                loop_function = None
            if logits_MemKG is not None:
                for i, query in enumerate(answers):  # for test the connection methods before or after the encoder
                    # q_emb[i]=tf.add(logits_MemKG, q_emb[i])
                    answers[i] = tf.concat(axis=1, values=[logits_MemKG, answers[i]])

            linear = rnn_cell_impl._linear
            batch_attn_size = array_ops.stack([batch_size, attn_size])
            attns = [array_ops.zeros(batch_attn_size, dtype=tf.float32) for _ in range(num_heads)]
            for a in attns:
                a.set_shape([None, attn_size])

            with tf.variable_scope("rnn_decoder"):
                single_cell_de = tf.nn.rnn_cell.GRUCell(self._rnn_neurons)
                cell_de = single_cell_de
                if self._num_layers > 1:
                    cell_de = tf.contrib.rnn.MultiRNNCell([single_cell_de] * self._num_layers)
                outputs = []
                prev = None
                #   pdb.set_trace()
                state = encoder_state
                for i, inp in enumerate(answers):
                    if loop_function is not None and prev is not None:
                        with tf.variable_scope("loop_function", reuse=True):
                            # We do not propagate gradients over the loop function.
                            # qichuan use the left half embedding (fact embedding) feed to self loop
                            #  fe=tf.split(1,2,inp)
                            #  inp_fa=fe[1]
                            #  inp_loop = array_ops.stop_gradient(loop_function(prev, i))
                            #  inp=tf.concat(1,[inp_fa,inp_loop])
                            inp = array_ops.stop_gradient(loop_function(prev, i))
                            if logits_MemKG is not None:
                                inp = tf.concat(axis=1, values=[logits_MemKG, inp])
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    inp = linear([inp] + attns, self._rnn_neurons, True)
                    #  inp=linear(inp,self._rnn_neurons,True)
                    output, state = cell_de(inp, state)
                    attns = attention(state)
                    #  pdb.set_trace()
                    with tf.variable_scope('AttnOutputProjecton'):
                        output = linear([output] + attns, self._vocab_size, True)
                    outputs.append(output)
                    if loop_function is not None:
                        prev = array_ops.stop_gradient(output)
                        # pdb.set_trace()
                        #	outputs_original=copy.copy(outputs)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
            return outputs  # ,outputs_original

    def _build_inputs(self,FLAGS):
        self._stories = tf.placeholder(tf.int32, [self._batch_size, self._memory_size, self._sentence_size],
                                       name="stories")
        self._queries = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name="queries")
        if FLAGS.model_type == 'memn2n':
            self._ans_fact=tf.placeholder(tf.int32, [self._batch_size,self._vocab_size],name="ans_fact")
        else:
            self._answers = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size + 1], name="answers")
            self._weight = tf.placeholder(tf.float32, [self._batch_size, self._sentence_size], name='A_weight')
        # self._lr = tf.placeholder(tf.float32, [], name="learning_rate")
            sign, answers_shifted = tf.split(self._answers, [1, -1], 1)
            self._answers_shifted = answers_shifted

    def _build_vars(self,FLAGS):
        if FLAGS.model_type == 'memn2n':
          with tf.variable_scope(self._name):
            init = tf.random_normal_initializer(stddev=0.1)
            A = init([self._vocab_size , self._embedding_size])
            C = init([self._vocab_size , self._embedding_size])
            #      self.reshape_to_rnn =self._init([int(self._rnn_neurons), self._embedding_size])

            self.A_1 = tf.Variable(A, name="A")

            self.C = []

            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name="C"))
        else:
            self.rnn_embedding = tf.get_variable("embedding", [self._vocab_size, self._rnn_neurons],
                                                         dtype=tf.float32)

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            # Use A_1 for thee question embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, queries)
            # pdb.set_trace()
            u_0 = tf.reduce_sum(q_emb * self._position_encoder, 1)
            u = [u_0]

            for hopn in range(self._hops):
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(self.A_1, stories)
                    m_A = tf.reduce_sum(m_emb_A * self._position_encoder, 2)
                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], stories)
                        m_A = tf.reduce_sum(m_emb_A * self._position_encoder, 2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                #		pdb.set_trace() # try to enlarge the probability to see the effect
                #	if (self.additional_info_size):
                #		u_1,p_2,p_3=tf.split(1,3,value=probs)#this function has version error
                #		p_3=p_3*5
                #		probs=tf.concat(1,[p_1,p_2,p_3])

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(self.C[hopn], stories)
                m_C = tf.reduce_sum(m_emb_C * self._position_encoder, 2)

                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Dont use projection layer for adj weight sharing
                # u_k = tf.matmul(u[-1], self.H) + o_k

                u_k = u[-1] + o_k

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                #	pdb.set_trace()
                #      return tf.matmul(u_k, tf.transpose(self.C[-1], [1,0])),tf.matmul(u_k,tf.transpose(self.reshape_to_rnn,[1,0]))
                return tf.matmul(u_k, tf.transpose(self.C[-1], [1, 0])), u_k

    def add_info_to_stories(self, stories, queries):
        """Runs the training algorithm over the passed batch
            loss: floating-point number, the loss computed for the batch
        """
        #	pdb.set_trace()
        vocab = self.vocab
        original_queries = []
        additional_info = []  # store the knowledge in english
        additional_triples = []
        checked_named_entities = set()
        for sents in queries:  # change the queries from code to english
            words = ''
            for word in sents:
                words = words + vocab.index_to_word(word) + ' '
            original_queries.append(words)
        for sents in original_queries:  # get relative knowledge from graph
            named_entities = get_continuous_chunks(sents)
            #		pdb.set_trace()
            #		for entity in named_entities: # delete repeated entity
            #			if entity in checked_named_entities:
            #				named_entities.remove(entity)
            #			else: checked_named_entities.add(entity)
            result = my_graph.read_rdf_kb(self._fb_map[1], named_entities)
            # if len(result)==0:
            #	pdb.set_trace()
            #	return stories
            #		pdb.set_trace()
            #	print result
            external_KB, addition_memory_triples = my_graph.get_additional_info(self._fb_map[0], result)
            additional_triples.append(addition_memory_triples)
            additional_info.append(external_KB)
            print(addition_memory_triples)
        # pdb.set_trace()

        # queries: entities: triples
        #	pdb.set_trace()
        #	change knowledge English to code for memory
        external_info = []  # for change knowledge to code
        for query_index, query_info in enumerate(additional_info):
            #		pdb.set_trace()
            temp = []
            for entity_info in query_info:
                for sent in entity_info:
                    #	sent=[x.strip() for x in re.split('\W+?', sent) if x.strip()]
                    sent = sent.split()
                    ls = max(0, self._sentence_size - len(sent))
                    temp.append([vocab.word_to_index(word) for word in sent] + [0] * ls)
                    #			pdb.set_trace()
            last_sequence = self.get_last_memory_sequence(stories, query_index)
            if not isinstance(last_sequence, (int)):
                print ("get worng sequence number for facts")
                pdb.set_trace()
            for i in range(len(temp)):
                #            		temp[i][-1] = self._vocab_size -self._memory_size - i-(self._memory_size-self.additional_info_size) + len(temp) # to sequence the facts
                temp[i][-1] = last_sequence + len(temp) - i - 1  # to continue sequence the facts

            external_info.append(temp)
        # pdb.set_trace()
        #	concate context and additional knowledge to memory
        stories = stories[:, 0:-self.additional_info_size]
        for idx, sents in enumerate(external_info):
            padding_sents_len = max(0, self.additional_info_size - len(sents))
            for _ in range(padding_sents_len):
                external_info[idx].append([0] * self._sentence_size)
                #	pdb.set_trace()
                #	if len(stories)!=len(external_info) or len(external_info[0])<1:
        if np.array(external_info).ndim < 3:
            print('external infor structure is wrong!')
            pdb.set_trace()
        # print('str:',stories.shape)
        #	print('ext:',np.array(external_info).shape)
        #	pdb.set_trace()
        stories = np.column_stack((stories, np.array(external_info)))
        stories = self.normalize_stories(stories)  # try see what happen if memory normalized
        self.question_memory(additional_triples,
                             stories)  # use addntional knowledge to create examples to train model by model itself
        #	pdb.set_trace()
        return stories

    def normalize_stories(self, stories):
        new_stories = []
        padding = []
        #	pdb.set_trace()

        for i in range(len(stories[0][0])):
            padding.append(0)
        for story in stories:
            new_story = []
            for lines in story:
                if np.sum(lines) > 0:
                    new_story.append(lines)
                    #		pdb.set_trace()
            last_index = new_story[-1][-1]
            for ind, line in enumerate(new_story):
                line[-1] = last_index + len(new_story) - ind - 1

            if len(new_story) < len(story):
                lp = len(story) - len(new_story)
                for i in range(lp):
                    new_story.append(padding)
            new_stories.append(new_story)
        # pdb.set_trace()
        return np.array(new_stories)

    def get_last_memory_sequence(self, stories, index):
        memory_query_index = stories[index]
        for line in memory_query_index:
            if np.sum(line) == 0:
                return last_line[-1]
            last_line = line
        return stories[index][-1][-1]

    def question_generate(self, eng_triples):
        #	queries=['Who is Mary Geneva Doud sibling','Where is John Colliani','Who is John Colliani in music','Who is instance Sandra Burnhard','Who is Daniel Conn','Who is artist Sandra Burnhard'] #for test
        #	queries=[['What is type instance Sandra Burnhard'],['Where is Sandra Burnhard','Where is Mary Geneva Doud'],['What is music recording artist Sandra Burnhard'],['Who is instance Sandra Burnhard'],['Who is Daniel Conn']]#,['Who is artist Sandra Burnhard']] #for test
        #	answers=[['award_winner'],['bedroom','bathroom'],['m'],['award_winner'],['measured_person']]#,['0rn_xs']] # for test
        queries = []
        answers = []
        #	pdb.set_trace()
        for triples in eng_triples:
            query = []
            answer = []
            if len(triples) > 0:
                if (len(triples[0]) != 0):
                    for subj, pred, obj in triples:
                        #				pdb.set_trace()
                        #		pred=self._cut_to_string(pred)
                        #		subj=self._cut_to_string(subj)
                        #		obj=self._cut_to_string(obj)
                        query.append('What is ' + subj + ' ' + pred)
                        if len(query) > self._sentence_size:
                            print (query)
                            pdb.set_trace()
                        if obj.find(' ') >= 0:
                            answer.append(obj[:obj.index(' ')])  # output is one word for test
                        else:
                            answer.append(obj)
                        query.append('What is ' + obj + ' ' + pred)
                        if subj.find(' ') >= 0:
                            answer.append(subj[:subj.index(' ')])
                        else:
                            answer.append(subj)
            queries.append(query)
            answers.append(answer)
        # pdb.set_trace()
        return queries, answers

    def _cut_to_string(self, word_list):
        pred_str = ''
        pred = [x.strip() for x in re.split('\W+?', word_list) if x.strip()]
        for x in pred: pred_str = pred_str + x + ' '
        return pred_str.strip()

    def question_memory(self, eng_triples, stories):
        s = []
        q = []
        a = []
        label = []
        #	pdb.set_trace()
        queries_batch, answers_batch = self.question_generate(eng_triples)

        if len(queries_batch) != len(stories):
            pdb.set_trace()
        for batch_index, queries in enumerate(queries_batch):

            if len(queries) == 0 or len(answers_batch[batch_index]) == 0:
                #			pdb.set_trace() # delete useless memory if no query are generated
                continue
            steps = 0  # sign the number of steps
            for index, query in enumerate(queries):
                query_words = query.split()
                answer_words = answers_batch[batch_index][index].split()
                if len(query) == 0 or len(answer_words) == 0:
                    # step over empty Q_A
                    steps += 1
                    continue
                label.append(self.vocab.word_to_index(answer_words[0]))
                lq = max(0, self._sentence_size - len(query_words))
                q_temp = [self.vocab.word_to_index(w) for w in query_words] + [0] * lq
                if len(q_temp) > self._sentence_size:
                    pdb.set_trace()
                q.append(q_temp)
                y = []
                la = max(0, self._sentence_size - len(answer_words))
                for word in answer_words:
                    y_vect = np.zeros(self._vocab_size)
                    y_vect[self.vocab.word_to_index(word)] = 1
                    y.append(y_vect)
                for temp in range(la):
                    y.append([0] * self._vocab_size)
                a.append(y)
            for i in range(len(
                    queries) - steps):  # expand memory length for same length queries and answers from question_generate
                #			pdb.set_trace()
                s.append(stories[batch_index])
        if len(s) != len(q): pdb.set_trace()
        #	s=np.array(s)
        #	pdb.set_trace()
        loss = self.introspection(s, q, a, 0.01)
        print ('self_loss', loss)
        i = 0
        while loss > 5:
            i = i + 1
            loss = self.introspection(s, q, a, 0.05)
            print(i, loss)
            if i > 0: break
        # self.reinforcement(s,q,label,a)
        #	pdb.set_trace()
        # second self-question process
        #	self.predict(s,q,pred_type='test')
        return loss

    def reinforcement(self, story, question, label, answer_idx):
        s = list(story)
        q = list(question)
        a = list(answer_idx)
        #	pdb.set_trace()
        predict_ops = self.predict(story, question, introspection=True)
        answer_idx = list()
        story = list()
        question = list()
        error_numb = []
        for index, predict in enumerate(predict_ops):
            if predict != label[index]:
                error_numb.append(index)
                #	error_numb.reverse()
        print('error list:', error_numb, ' total:', len(predict_ops))
        #	pdb.set_trace()
        for numb in error_numb:
            story.append(s[numb])
            question.append(q[numb])
            answer_idx.append(a[numb])
        # pdb.set_trace()
        if len(story) > self._batch_size:
            for x in range(10):
                loss = self.introspection(story, question, answer_idx, 0.02)
                print ('loss', loss)
                #	pdb.set_trace()
                #	self.reinforcement(s,q,label,a) # seems useless, you can set parameter pred_type of predict to True to get same affection
        return

    def introspection(self, stories, queries, answers, ans_w, learning_rate):
        total_loss = 0.0
        for start in range(0, len(stories), self._batch_size):
            end = start + self._batch_size
            #	pdb.set_trace()
            if end > len(stories):
                for padding_no in range(end - len(stories)):
                    stories.append(np.array(len(stories[0]) * [[0] * len(stories[0][0])]))
                    queries.append(np.array(([0] * len(queries[0]))))
                    answers.append(np.array(len(answers[0]) * [[0] * len(answers[0][0])]))
                    #	pdb.set_trace()
            s = np.array(stories[start:end])
            q = np.array(queries[start:end])
            a = np.array(answers[start:end])
            feed_dict = {self._stories: s, self._queries: q, self._answers: a, self._lr: learning_rate}
            loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
            total_loss += loss
        return total_loss

    def predict(self, stories, queries, answers=None, weight=None,fact=None, process_type='train', lr=None, introspection=False):
        """Predicts answers
        """
        if introspection:
            introspection_pred = []
            for start in range(0, len(stories), self._batch_size):
                end = start + self._batch_size
                if end > len(stories): continue  # throw away data less than a batch
                s = stories[start:end]
                q = queries[start:end]
                feed_dict = {self._stories: s, self._queries: q}
                pred = self._sess.run(self.predict_op, feed_dict=feed_dict)
                introspection_pred += list(pred)
            return introspection_pred
        if process_type == 'test':
            #		new_stories=self.add_info_to_stories(stories, queries)
            #		pdb.set_trace()
            #		stories=new_stories
            self.type = 'test'
            if fact is None:
                feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._weight: weight}
                return self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict), self.vocab
            else:
                feed_dict = {self._stories: stories, self._queries: queries, self._ans_fact:fact}
                return self._sess.run([self.loss_op, self.predict_key_fact], feed_dict=feed_dict), self.vocab

        if process_type == 'train':
            if fact is None:
                feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers,
                             self._weight: weight}
            else:
                feed_dict = {self._stories: stories, self._queries: queries,self._ans_fact:fact}
            return self._sess.run([self.loss_op, self.train_op, self.loss_summary], feed_dict=feed_dict)
        if process_type == 'valid':
            if fact is None:
                feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._weight: weight}
                return self._sess.run([self.loss_op, self.predict_op], feed_dict=feed_dict)
            else:
                feed_dict = {self._stories: stories, self._queries: queries, self._ans_fact:fact}
                return self._sess.run([self.loss_op, self.predict_key_fact], feed_dict=feed_dict)
        print('Error _______ :invalid process_type')
