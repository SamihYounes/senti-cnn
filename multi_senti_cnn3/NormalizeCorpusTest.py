# -*- coding: utf-8 -*-
#usage: python py_normalize_corpus.py Tweets.txt
import re
import io
import sys
import random
import codecs
import numpy as np

def segmentText(str_text):
	str_text = re.sub(r'([^a-zA-ZÄäÖöÜüßáàÀâåäãçÇéÉèÈêÊëËíîÎïòôÔöõœŒrśúùû])([a-zA-ZÄäÖöÜüßáàÀâåäãçÇéÉèÈêÊëËíîÎïòôÔöõœŒrśúùû])', r'\1 \2', str_text)
	str_text = re.sub(r'([a-zA-ZÄäÖöÜüßáàÀâåäãçÇéÉèÈêÊëËíîÎïòôÔöõœŒrśúùû])([^a-zA-ZÄäÖöÜüßáàÀâåäãçÇéÉèÈêÊëËíîÎïòôÔöõœŒrśúùû])', r'\1 \2', str_text)
	str_text = re.sub(r'([^0-9])([0-9])', r'\1 \2', str_text)
	str_text = re.sub(r'([0-9])([^0-9])', r'\1 \2', str_text)
	str_text = re.sub(r' +', r' ', str_text)
	return str_text
    
def UpdateWordlist(str_text, word_dict_freq, word_dict_seed, seed, word_list):
    words = str_text.split(' ')
    for word in words:
        if word in word_dict_freq:
            word_dict_freq[word] += 1
        else:
            word_dict_freq[word] = 1
            word_list += [word]
            seed += 1
            word_dict_seed[word] = seed
    return word_dict_freq, word_dict_seed, seed, word_list

def digitizeSentences(sentence_list, word_dict_seed, word_dict_freq):
    for i, sentence in enumerate(sentence_list):
        new_sentence = []
        for word in sentence[0].split(' '):
            if word in word_dict_seed and word_dict_freq[word] > 3 and word_dict_freq[word] < len(word_dict_freq) - 3:
                word = word_dict_seed[word]
            else:
                word = 2
            new_sentence.append(word)
        sentence_list[i] = new_sentence
    return sentence_list

def semDBNormalize(semfiles):
    if isinstance(semfiles, str):
        semfiles = [semfiles]
    if len(semfiles) > 4:
        print "Max 4 data files supported"
        exit()
    word_dict_freq = {}
    word_dict_seed = {}
    word_list = []
    word_list += ["UNKNOWN"]
    word_list += ["PADDING"]
    sentences = list(x for x in range(len(semfiles)))
    polarity = list(x for x in range(len(semfiles)))
    seed = 2
    for i, semfile in enumerate(semfiles):
        fread = io.open(semfile,'r',encoding='utf8')
        lines = fread.read()
        line_list = lines.split('\n')
        SEED = 448
        random.seed(SEED)
        #if "train" in semfile.lower():
        #    print "SHUFFLING ... " + semfile
        random.shuffle(line_list)
        sentences[i] = []
        polarity[i] = []
        for lineNo, str_text in enumerate(line_list):
            neg = 0
            neg_rep = 3
            pos = 0
            pos_rep = 3
            if lineNo % 1000 == 0 and lineNo != 0:
                print "loading %s data line ... %d" % (semfile, lineNo)
            str_text = str_text.strip()
            if not str_text:
                continue
            if '\t' in str_text:
                cols = str_text.split('\t')
                if "xtest" in semfile:
                    str_text = cols[0]
                    polarity[i].append(0)
                else:
                    str_text = cols[0]
                    label = cols[1]
                    if int(label) == 0:
                        neg = 1
                    if int(label) == 1:
                        pos = 1
                    polarity[i].append(int(label))
                    if pos == 1 and "train" in semfile.lower():
                        for n in range(pos_rep):
                            polarity[i].append(int(label))
                    if neg == 1 and "train" in semfile.lower():
                        for n in range(neg_rep):
                            polarity[i].append(int(label))
            else:
                print "str_text:", str_text
                print("wrong format in %s line %d, must be tab separated with text and polarity (0 for neg and 1 for pos)" % (semfile, lineNo))
                #exit()
                continue
        
            str_text = re.sub(u'\n', '', str_text)
            str_text = re.sub(u'\r', '', str_text)
            # remove cashida
            str_text = re.sub(r'ـ', '', str_text)
            # remove links
            str_text = re.sub(r'http://.*?( |\t|$)', '', str_text)
            str_text = segmentText(str_text)
            str_text = re.sub(r'(\D)(\1{2,})', r'\1\1', str_text)
            #rem duplicate characters
            str_text = str_text.strip()
            sentences[i].append([str_text])
            if pos == 1 and "train" in semfile.lower():
                #print "oversampling pos"
                for n in range(pos_rep):
                    sentences[i].append([str_text])
            if neg == 1 and "train" in semfile.lower():
                #print "oversampling neg"
                for n in range(neg_rep):
                    sentences[i].append([str_text])
            word_dict_freq, word_dict_seed, seed, word_list = UpdateWordlist(str_text, word_dict_freq, word_dict_seed, seed, word_list)
    print "Number of lines:", len(sentences[i])
    for i, k in enumerate(sentences):
        sentences[i] = digitizeSentences(sentences[i], word_dict_seed, word_dict_freq)
    vocabsize = len(word_dict_freq)
    print "Number of words:", len(word_dict_freq)
        
    #return word_dict_freq, word_dict_seed, sentences, polarity
    seed += 1
    if len(semfiles) == 1:
        split_c = int(len(sentences[0]) * .8)
        return (sentences[0][0:split_c], polarity[0][0:split_c]), (sentences[0][split_c:], polarity[0][split_c:]), seed
    if len(semfiles) == 2:
        return (sentences[0], polarity[0]), (sentences[1], polarity[1]), seed
    if len(semfiles) == 3:
        return (sentences[0], polarity[0]), (sentences[1], polarity[1]), (sentences[2], polarity[2]), seed
    if len(semfiles) == 4:
        return (sentences[0], polarity[0]), (sentences[1], polarity[1]), (sentences[2], polarity[2]), (sentences[3], polarity[3]), seed
