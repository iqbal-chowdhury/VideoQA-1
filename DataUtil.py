import numpy as np
import os
import MovieQA_benchmark as MovieQA
import re
import h5py
import math
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')
snowball = SnowballStemmer('english') 

def preprocess_sentence(line):
	'''strip all punctuation, keep only alphanumerics
	'''
	line = re_alphanumeric.sub('', line)
	line = re_multispace.sub(' ', line)
	return line


def create_vocabulary(QAs, stories, word_thresh=2, v2i={'': 0, 'UNK':1}):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	'''
	print 'Create vocabulary...'

	# Get all story words
	all_words = [word for story in stories for sent in story for word in sent]
	print('number of words: %d' %len(all_words))

	# Parse QAs to get actual words 
	# 		&& Append question and answer words to all_words
	# QA_words = []
	# for QA in QAs:
	# 	QA_words.append({})
	# 	QA_words[-1]['q_w'] = preprocess_sentence(QA.question.lower().split(' '))
	# 	QA_words[-1]['a_w'] = [preprocess_sentence(answers.lower()).split(' ') for answer in QA.answers]

	
	# for QAw in QA_words:
	# 	all_words.extend(QAw['q_w'])
	# 	for answer in QAw['a_w']:
	# 		all_words.extend(answer)

	QA_words = {}
	for QA in QAs:
		temp = {}
		q_w = preprocess_sentence(QA.question.lower()).split(' ')
		a_w = [preprocess_sentence(answer.lower()).split(' ') for answer in QA.answers]
		temp['q_w'] = q_w
		temp['a_w'] = a_w
		temp['qid'] = QA.qid
		temp['imdb_key'] = QA.imdb_key
		temp['question'] = QA.question
		temp['answers'] = QA.answers
		temp['correct_index'] = QA.correct_index
		# temp['plot_alignment'] = QA.plot_alignment
		temp['video_clips'] = QA.video_clips

		
		QA_words[QA.qid]=temp

		all_words.extend(q_w)
		for answer in a_w:
			all_words.extend(answer)


	# threshold vocabulary, at least N instances of every word
	vocab = Counter(all_words)
	vocab = [k for k in vocab.keys() if vocab[k] >= word_thresh]

	# create vocabulary index
	for w in vocab:
		if w not in v2i.keys():
			v2i[w] = len(v2i)
	
	print('Created a vocabulary of %d words. Threshold removed %.2f %% words'\
		%(len(v2i), 100*(1. * len(set(all_words))-len(v2i))/len(all_words)))

	return QA_words, v2i


def S2I(sentence, v2i, fixed_len):
	'''
		len_qa: fixed length of question or answer
	'''
	res = []
	for idx, w in enumerate(sentence):
		if idx<fixed_len:
			if w in v2i.keys():
				res.append(v2i[w])
			else:
				res.append(v2i['UNK'])
	while(len(res)<fixed_len):
		res.append(v2i[''])
	return res


def getBatchIndexedQAs(batch_qid_list,QA_words,v2i, nql=16, nqa=10, numOfChoices=2):
	'''
		batch_qid_list: list of qid
		QA_words: all the QAs, contains question words and answer words
		v2i: vocabulary to index
		nql: length of question
		nqa: length of answer
		numOfChoices: number of Choices utilized per QA, default set to 2 ==> right/wrong

		return: questions, answers
			both of them are numeric indexed
	'''
	assert numOfChoices==2

	batch_size = len(batch_qid_list)
	questions = np.zeros((batch_size,nql),dtype='int32')
	answers = np.zeros((batch_size,numOfChoices,nqa),dtype='int32')

	for idx, qid in enumerate(batch_qid_list):
		# set question 
		
		questions[idx][:]=S2I(QA_words[qid]['q_w'], v2i,nql)
		# set anwsers
		ground_answer_pos = np.random.randint(0,numOfChoices)

		
		# set correct answer
		correct_index = int(QA_words[qid]['correct_index'])
		answers[idx][ground_answer_pos][:] = S2I(QA_words[qid]['a_w'][correct_index], v2i, nqa)

		wrong_index = np.random.randint(0,5)
		while(wrong_index==correct_index):
			wrong_index = np.random.randint(0,5)

		# set correct answer
		# if numOfChoices != 2 , the following code is wrong 
		answers[idx][1-ground_answer_pos][:]=S2I(QA_words[qid]['a_w'][wrong_index], v2i, nqa)

	return questions,answers

def getBatchVideoFeature(batch_qid_list, QA_words, hf, feature_shape):

	batch_size = len(batch_qid_list)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	timesteps = feature_shape[0]

	for idx, qid in enumerate(batch_qid_list):
		video_clips = QA_words[qid]['video_clips']
		imdb_key = QA_words[qid]['imdb_key']
		feature = []
		if len(video_clips) != 0:
			for clip in video_clips:
				feature.extend(hf['/'+video_clips[0]+'/'+video_clips[0]][:]) # feature.shape
			print(len(feature))

			interval = int(math.floor((len(feature)-1)/(timesteps-1)))

			input_video[idx] = feature[1::interval][0:timesteps]
	return input_video


def main():
	mqa = MovieQA.DataLoader()
	# get 'video-based' QA task training set
	vl_qa, _ = mqa.get_video_list('train', 'qa_clips')  # key: 'train:<id>', value: list of related clips
	# vl_qa, _ = mqa.get_video_list('train', 'all_clips') # key:moive vid, value:list of related movid all_clips
	
	stories, QAs = mqa.get_story_qa_data('train', 'subtitle')

	# Create vocabulary
	QA_words, v2i = create_vocabulary(QAs, stories, word_thresh=2, v2i={'': 0, 'UNK':1})


	
	all_video_train_list = vl_qa.keys()

	batch_size = 20
	total_train_qa = len(all_video_train_list)
	num_batch = int(round(total_train_qa*1.0/batch_size))

	total_epoch = 100

	hf = h5py.File('/home/wb/myfile.hdf5','r')
	feature_shape = (10,1024)
	for epoch in xrange(total_epoch):
		#shuffle
		all_video_train_list =  np.random.permutation(all_video_train_list)
		for batch_idx in xrange(num_batch):
			batch_qa = all_video_train_list[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_qa)]
			questions,answers = getBatchIndexedQAs(batch_qa,QA_words,v2i, nql=16, nqa=10, numOfChoices=2)
			input_video = getBatchVideoFeature(batch_qa, QA_words, hf, feature_shape)
			print(input_video)
			break
		break


if __name__=='__main__':
	main()