# -*- coding: utf-8 -*-
import csv
import json
import random
import re
from PIL import Image
import numpy as np
import pandas as pd
# import spacy
from gensim.models import KeyedVectors # load word2vec
import torch.utils.data as data
import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa")
sys.path.append(root_path)
import config # pre-process image

KEY_SEPARATOR = '/'


class VQADataProvider:

    def __init__(self, opt, mode='train'):
        self.opt = opt  # see config.py
        self.batchsize = opt.BATCH_SIZE
        self.n_skipped = None  # count of problematic instance
        self.qid_list = None   # List of QA pair ID
        self.batch_index = None
        self.batch_len = None
        self.epoch_counter = None
        self.rev_adict = None  # {{"String"}: ID in answer dictionary}
        self.q_max_length = opt.MAX_WORDS_IN_QUESTION
        self.a_max_length = opt.MAX_WORDS_IN_ANSWER
        self.mode = mode
        self.qdic, self.adic, _ = VQADataProvider.load_data(mode)
        self.embedding_path = opt.EMBEDDING_PATH
        self.MED_EMBEDDING_SIZE = 200  # this value will automatically update once embedding is loaded
        # self.transform = config.data_transforms['%s' % mode]  # pre-processing images
        self.transform = config.transform  # pre-processing images

        # load question vocab and answer vocab
        # folder = '/home/tryn/MedVQA/Log/%s/%s' % (opt.MODEL_NAME, opt.TRAIN_DATA_SPLITS)
        # with open('%s/question_vocab.json' % folder, 'r') as f:
        #     self.question_vocab = json.load(f)  # questions vocabulary dictionary {(<'word'>, <index>}
        # with open('%s/answer_vocab.json' % folder, 'r') as f:
        #     self.answer_vocab = json.load(f)  # answers vocabulary dictionary {(<'word'>, <index>}
        folder = os.path.join(root_path, "vocab")
        with open("%s/question.json" % folder, "r") as f:
            self.question_vocab = json.load(f)
        with open("%s/answer.json" % folder, "r") as f:
            self.answer_vocab = json.load(f)

        self.n_ans_vocabulary = len(self.answer_vocab)
        # self.nlp = spacy.load('en_vectors_web_lg')
        # self.MED, self.MED_EMBEDDING_SIZE = VQADataProvider.loadMED_Embedding(self.embedding_path)
        self.MED = VQADataProvider.loadMED_Embedding(self.embedding_path)
        # MED is a pre-trained medical word embedding. Note the values are np arrays not lists for quick loading

    @staticmethod
    def loadMED_Embedding(embedding_path):
        print("Loading MED Embedding ......")
        # f = open(embedding_path, 'r')
        # MED = {}
        # for line in f:
        #     splitLine = line.split()
        #     word = splitLine[0]
        #     embedding = np.array([float(val) for val in splitLine[1:]])
        #     MED[word] = embedding
        # print("Done." + str(len(MED)) + " words loaded!")
        # MED_EMBEDDING_SIZE = len(MED[MED.keys()[0]])
        MED = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        print("Done." + str(len(MED.wv.vocab)) + " words loaded!")
        return MED

    @staticmethod
    def load_vqa_csv(data_split):
        """
        Parses the question and answer csv files for the given data split.
        :param data_split= 'train'/ 'val'/ 'test'
        :return question dictionary and answer dictionary
        """
        qdic, adic, qa_pair = {}, {}, {}
        # assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
        # with open(config.DATA_PATHS[data_split]['QA_file']) as csvfile:
        with open(os.path.join(root_path, data_split, "VQAMed2018%s-QA.csv"%data_split)) as csvfile:
            QA = csv.reader(csvfile, delimiter='\t', quotechar='\n')
            for rows in QA:
                qdic[data_split + KEY_SEPARATOR + str(rows[0])] = \
                    {'qstr': rows[2], 'iid': rows[1]}
                if 'test' not in data_split:
                    if len(rows) != 4:
                        # Ideally, rows = [<qa_id>, <image_id>, <question>, <answer>]
                        print('Row ' + str(rows[0]) + ' is not in QA form:' + str(rows))
                    adic[data_split + KEY_SEPARATOR + str(rows[0])] = \
                        {'astr': rows[3], 'iid': rows[1]}
                    qa_pair[data_split + KEY_SEPARATOR + str(rows[0])] = \
                        {'qastr': rows[2] + ' ' + rows[3], 'iid': rows[1]}
        print('parsed ' + str(len(qdic)) + ' Q&A instances for ' + str(data_split))
        return qdic, adic, qa_pair

    @staticmethod
    def load_data(data_split_str):
        """
        :param data_split_str: 'train','val', 'test', 'train+val'
        :return: combined question dictionary and answer dictionary
        """
        all_qdic, all_adic, all_qa_pair = {}, {}, {}
        for data_split in data_split_str.split('+'):
            # assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            qdic, adic, qa_pair = VQADataProvider.load_vqa_csv(data_split)
            all_qdic.update(qdic)
            all_adic.update(adic)
            all_qa_pair.update(qa_pair)
        return all_qdic, all_adic, all_qa_pair

    def getQuesIds(self):
        return list(self.qdic.keys())

    @staticmethod
    def getStrippedQuesId(instance_id):
        return instance_id.split(KEY_SEPARATOR)[1]

    def getImgId(self, instance_id):
        return self.qdic[instance_id]['iid']

    def getQuesStr(self, instance_id):
        return self.qdic[instance_id]['qstr']

    def getAnsObj(self, instance_id):
        if self.mode == 'test':
            return -1
        return self.adic[instance_id]['astr']

    @staticmethod
    def loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\/']:
            t_str = re.sub(i, ' ', t_str)
        for i in [r'\.', r'\;', r'\?']:
            t_str = re.sub(i, ' <BREAK> ', t_str)
        for i in [r'\-']:
            t_str = re.sub(i, '', t_str)
        q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
        q_list = ['<START>'] + q_list + ['<END>']
        return q_list


    @staticmethod
    def qlist_to_matrix(q_list, max_length, EMB, embedding_size=200):
        ''' 
        convert question string to embedding matrix
        '''
        MED_matrix = np.zeros((max_length, embedding_size))
        for i in range(max_length):
            if i >=len(q_list):
                break
            else:
                w = q_list[i]
                if w not in EMB.wv.vocab:
                    MED_matrix[i] = np.zeros(embedding_size)
                else:
                    MED_matrix[i] = EMB[w]
        return MED_matrix

    @staticmethod
    def alist_to_vec(a_list, length, ans_vocab):
        a_vec = np.zeros(length)
        for w in a_list:
            if w not in ans_vocab:
                w = "<UNKNOWN>"
            a_vec[ans_vocab[w]] += 1
        return a_vec/np.sum(a_vec)


    def qlist_to_vec(self, q_list, qatype='question'):
        """
        Converts a list of words into a format suitable for the embedding layer.
        :param
        max_length -- the maximum length of a word sequence
        q_list -- a list of words which are the tokens in the question
        qatype -- the input list pf words are from questions or answers.
        :returns
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector: padded --> 0; non-padded --> 1
        """
        if qatype == 'question':
            max_length = self.q_max_length
        else:
            max_length = self.a_max_length
        qvec = np.zeros(max_length)
        cvec = np.zeros(max_length)
        MED_matrix = np.zeros((max_length, self.MED_EMBEDDING_SIZE))
        """  pad on the left   """
        # for i in xrange(max_length):
        #     if i < max_length - len(q_list):
        #         cvec[i] = 0
        #     else:
        #         w = q_list[i-(max_length-len(q_list))]
        #         # is the word in the vocabulary?
        #         if self.vdict.has_key(w) is False:
        #             w = ''
        #         qvec[i] = self.vdict[w]
        #         cvec[i] = 0 if i == max_length - len(q_list) else 1
        """  pad on the right   """
        for i in range(max_length):  # if the words are
            if i >= len(q_list):
                # pass
                break
            else:
                w = q_list[i]
                # if self.MED.has_key(w) is False:  # beyond embedding
                if w not in self.MED.wv.vocab:
                    # self.MED[w] = self.nlp(u'%s' % w).vector  # use embeddings from spacy to handle out-of-embedding (check the dimension first)
                    # self.MED[w] = np.zeros(self.MED_EMBEDDING_SIZE)  # initial the out-of-embedding with zeros
                    MED_matrix[i] = np.zeros(self.MED_EMBEDDING_SIZE)
                else:
                    MED_matrix[i] = self.MED[w]
                cvec[i] = 1
                if qatype == 'question':
                    # if self.question_vocab.has_key(w) is False:  # beyond vocabulary
                    if w not in self.question_vocab:
                        # w = ''  # ignore Oov
                        print('"%s" is out of %s vocabulary' % (w, qatype))
                        w = '<UNKNOWN>'  # label Oov as <UNKNOWN>
                    qvec[i] = self.question_vocab[w]  # words out of question vocabulary will be indexed with 0
                else:   # look up into the answer vocabulary dictionary
                    # if self.answer_vocab.has_key(w) is False:  # beyond vocabulary
                    if w not in self.answer_vocab:
                        print('"%s" is out of %s vocabulary' % (w, qatype))
                        # w = ''  # ignore Oov
                        w = '<UNKNOWN>'  # label Oov as <UNKNOWN>
                    qvec[i] = self.answer_vocab[w]  # words out of question vocabulary will be indexed with 0

        return qvec, cvec, MED_matrix

    def vec_to_answer(self, ans_symbol):  # input is an index not a list of indices
        """ Return answer id if the answer is included in vocabulary otherwise 0 """
        if self.rev_adict is None:
            rev_adict = {}
            for k, v in self.answer_vocab.items():  # flip key-value pairs
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]

    def create_batch(self, instance_id_list):
        # question matrices
        qvec = np.zeros((self.batchsize, self.q_max_length))
        q_cvec = np.zeros((self.batchsize, self.q_max_length))
        q_MED_matrix = np.zeros((self.batchsize, self.q_max_length, self.MED_EMBEDDING_SIZE))
        # answer matrices
        avec = np.zeros((self.batchsize, self.a_max_length))
        a_cvec = np.zeros((self.batchsize, self.a_max_length))
        a_MED_matrix = np.zeros((self.batchsize, self.a_max_length, self.MED_EMBEDDING_SIZE))

        # image matrices
        ivec = np.zeros((self.batchsize, 3, 224, 224))  # use the same transformation procedure that end up with 3(RGB)*224*224 images

        for i, instance_id in enumerate(instance_id_list):

            # load raw question information
            q_str = self.getQuesStr(instance_id)  # question string
            a_str = self.getAnsObj(instance_id)  # answer string
            i_id = self.getImgId(instance_id)  # image id

            # convert question to vec
            q_list = VQADataProvider.seq_to_list(q_str)
            a_list = VQADataProvider.seq_to_list(a_str)
            t_qvec, t_q_cvec, t_q_MED_matrix = self.qlist_to_vec(q_list, qatype='question')
            t_avec, t_a_cvec, t_a_MED_matrix = self.qlist_to_vec(a_list, qatype='answer')

            t_img = self.loader(os.path.join(root_path, self.mode, 'VQAMed2018train-images', '%s.jpg' % i_id))
            if self.transform is not None:
                t_ivec = self.transform(t_img)
            else:
                t_ivec = np.zeros(224, 224)
            # print(instance_id)
            qvec[i, ...] = t_qvec
            q_cvec[i, ...] = t_q_cvec
            q_MED_matrix[i, ...] = t_q_MED_matrix
            avec[i, ...] = t_avec
            a_cvec[i, ...] = t_a_cvec
            a_MED_matrix[i, ...] = t_a_MED_matrix
            ivec[i, ...] = t_ivec

        return qvec, q_cvec, q_MED_matrix, avec, a_cvec, a_MED_matrix, ivec

    def get_batch_vec(self):
        if self.batch_len is None:
            self.n_skipped = 0  # problematic instance is counted by n_skipped
            instance_id_list = self.getQuesIds()
            random.shuffle(instance_id_list)
            self.qid_list = instance_id_list
            self.batch_len = len(instance_id_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def instance_check(instance_id):  # already cleaned the dataset so its all set
            return True

        counter = 0
        t_inst_id_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test':
                t_inst_id_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif instance_check(t_qid):  # model = 'train'
                t_inst_id_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1

            if self.batch_index < self.batch_len - 1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1  # all instances have been used
                instance_id_list = self.getQuesIds()
                random.shuffle(instance_id_list)
                self.qid_list = instance_id_list
                self.batch_index = 0
                print("%d questions were skipped in epoch %s" % (self.n_skipped, self.epoch_counter))
                self.n_skipped = 0

        t_batch = self.create_batch(t_inst_id_list)
        return t_batch + (t_inst_id_list, t_iid_list, self.epoch_counter)


class VQADataset(data.Dataset):
    """
    provide input data for pytorch dataloader
    """

    # def __init__(self, mode, opt, folder=None):
    #     self.batchsize = opt.BATCH_SIZE
    #     self.mode = mode
    #     self.folder = folder
    #     self.max_length = opt.MAX_WORDS_IN_QUESTION
    #     if self.mode == 'val' or self.mode == 'test':
    #         pass
    #     else:
    #         self.dp = VQADataProvider(opt, mode=self.mode)

    # def __getitem__(self, idx):
    #     if self.mode == 'val' or self.mode == 'test':
    #         qvec, q_cvec, q_MED_matrix, avec, a_cvec, a_MED_matrix, ivec, epoch = None, None, None, None, None, None, None, None
    #         pass
    #     else:
    #         qvec, q_cvec, q_MED_matrix, avec, a_cvec, a_MED_matrix, ivec, _, _, epoch = self.dp.get_batch_vec()
    #     q_length = np.sum(q_cvec, axis=1)
    #     a_length = np.sum(a_cvec, axis=1)
    #     return qvec, q_length, q_MED_matrix, avec, a_length, a_MED_matrix, ivec, epoch

    # def __len__(self):
    #     return 150000  # size of dataset. Haven't decided
    def __init__(self, mode, opt):
        self.opt = opt
        self.EMB = VQADataProvider.loadMED_Embedding(self.opt.EMBEDDING_PATH)
        self.EMBEDDING_SIZE = 200
        self.mode = mode
        # self.qdic, self.adic, _ = VQADataProvider.load_data(mode)
        # # qdic: {"train/indexInQAfile": {'qstr': question, 'iid': image_file_name}}
        # # adic: {"train/indexInQAfile": {'astr': answer, 'iid': image_file_name}}
        self.q_a_i_df = pd.read_csv(os.path.join(root_path, self.mode, "VQAMed2018%s-QA.csv"%self.mode), delimiter="\t", header=None, names=["index", "img_id", "question", "answer"])[["img_id", "question", "answer"]]
        # # shuffle the data
        # self.q_a_i_df = self.orig_df.sample(frac=1.0, replace=False, random_state=None)
        self.img_ids = self.q_a_i_df["img_id"]
        self.questions = self.q_a_i_df["question"]
        self.answer = self.q_a_i_df["answer"]
        # later, write the transform function in data provider script and use more complex transformation
        self.transform = config.transform
        self.data_len = self.q_a_i_df.shape[0]
        with open(os.path.join(root_path, "vocab/answer.json"), "r") as f:
            self.answer_vocab = json.load(f)
        self.img_foder = os.path.join(root_path, self.mode, "VQAMed2018%s-images"%self.mode)


    def __getitem__(self, index):
        # img_matrix = np.zeros((3, 224, 224))
        answer_vec = np.zeros(self.opt.NUM_OUTPUT_UNITS)

        # get question matrix 
        q_list = VQADataProvider.seq_to_list(self.questions[index])
        q_EMB_matrix = VQADataProvider.qlist_to_matrix(q_list, self.opt.MAX_WORDS_IN_QUESTION, self.EMB, embedding_size=self.EMBEDDING_SIZE)

        # get image matrix (3 x 224 x 224)
        image_path = os.path.join(root_path, self.mode, 'VQAMed2018%s-images'%self.mode, '%s.jpg'%self.img_ids[index])
        image = Image.open(image_path).convert('RGB')
        img_matrix = self.transform(image)

        # answer vector, length is 3000, which is the length of the output of the model.
        # each value means the probability of the word with the same index appearing in one answer
        a_list = VQADataProvider.seq_to_list(self.answer[index])
        a_vec = VQADataProvider.alist_to_vec(a_list, self.opt.NUM_OUTPUT_UNITS, self.answer_vocab)

        return q_EMB_matrix, img_matrix, a_vec

    def __len__(self):
        return self.data_len



