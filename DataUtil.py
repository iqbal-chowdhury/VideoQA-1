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


def S2I(sen, v2i, fixed_len):
	'''
		len_qa: fixed length of question or answer
	'''
	sentence = preprocess_sentence(sen.lower()).split(' ')
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


def getBatchIndexedQAs(batch_qas_list,QA_words,v2i, nql=16, nqa=10, numOfChoices=2, phase='train'):
	'''
		batch_qas_list: list of qas
		QA_words: all the QAs, contains question words and answer words
		v2i: vocabulary to index
		nql: length of question
		nqa: length of answer
		numOfChoices: number of Choices utilized per QA, default set to 2 ==> right/wrong

		return: questions, answers, ground_truth
			both of them are numeric indexed
			ground_truth is one hot vector
	'''

	batch_size = len(batch_qas_list)
	questions = np.zeros((batch_size,nql),dtype='int32')
	answers = np.zeros((batch_size,numOfChoices,nqa),dtype='int32')
	ground_truth = np.zeros((batch_size,numOfChoices),dtype='int32')

	for idx, qa in enumerate(batch_qas_list):
		# set question 
		qid = qa.qid
		questions[idx][:]=S2I(qa.question, v2i,nql)
		
		
		# set anwsers
		if numOfChoices==2:
			ground_answer_pos = np.random.randint(0,numOfChoices)
			ground_truth[idx][ground_answer_pos]=1
			
			# set correct answer
			correct_index = int(qa.correct_index)
			answers[idx][ground_answer_pos][:] = S2I(qa.answers[correct_index], v2i, nqa)



			wrong_index = np.random.randint(0,5)
			while(wrong_index==correct_index):
				wrong_index = np.random.randint(0,5)

			# set wrong answer
			answers[idx][1-ground_answer_pos][:]=S2I(qa.answers[wrong_index], v2i, nqa)
		elif numOfChoices==5:
			
			# set correct answer
			correct_index = int(qa.correct_index)
			ground_truth[idx][correct_index]=1
			for ans_idx, ans in enumerate(qa.answers):
				answers[idx][ans_idx][:]=S2I(ans, v2i, nqa)

		else:
			raise ValueError('Invalid numOfChoices: ' + numOfChoices)

	return questions,answers,ground_truth

def getBatchVideoFeature(batch_qas_list, QA_words, hf, feature_shape):
	'''
		video-based QA
		there are video clips in all QA pairs.  
	'''

	batch_size = len(batch_qas_list)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	timesteps = feature_shape[0]

	for idx, qa in enumerate(batch_qas_list):
		qid = qa.qid
		video_clips = qa.video_clips
		imdb_key = qa.imdb_key
		clips_features = []
		if len(video_clips) != 0:
			for clip in video_clips:
				dataset = imdb_key+'/'+clip
				if imdb_key in hf.keys() and clip in hf[imdb_key].keys():
					clips_features.extend(hf[dataset][:]) # clips_features.shape
			# print(idx,qid,len(clips_features))

			if(len(clips_features)<=0):
				# if there are not vlid features
				for clip in hf[imdb_key].keys():
					dataset = imdb_key+'/'+clip
					clips_features.extend(hf[dataset][:]) # clips_features.shape

			
			if(len(clips_features)>=timesteps):
				interval = int(math.floor((len(clips_features)-1)/(timesteps-1)))
				input_video[idx] = clips_features[0::interval][0:timesteps]
			else:
				input_video[idx][:len(clips_features)] = clips_features
				for last_idx in xrange(len(clips_features),timesteps):
					input_video[idx][last_idx]=clips_features[-1]

	return input_video


def main():
	
	task = 'video-based' # video-based or subtitle-based

	mqa = MovieQA.DataLoader()


	# get 'subtitile-based' QA task dataset
	stories, subtitle_QAs = mqa.get_story_qa_data('train', 'subtitle')

	# Create vocabulary
	QA_words, v2i = create_vocabulary(subtitle_QAs, stories, word_thresh=2, v2i={'': 0, 'UNK':1})

	# get 'video-based' QA task training set
	vl_qa, video_QAs = mqa.get_video_list('train', 'qa_clips')  # key: 'train:<id>', value: list of related clips
	# vl_qa, _ = mqa.get_video_list('train', 'all_clips') # key:moive vid, value:list of related movid all_clips


	
	all_video_train_list = video_QAs

	batch_size = 20
	total_train_qa = len(all_video_train_list)
	num_batch = int(round(total_train_qa*1.0/batch_size))

	total_epoch = 100

	hf = h5py.File('/home/wb/movie_feature.hdf5','r')
	feature_shape = (10,1024)
	for epoch in xrange(total_epoch):
		#shuffle
		np.random.shuffle(all_video_train_list)
		for batch_idx in xrange(num_batch):
			batch_qa = all_video_train_list[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_qa)]
			questions,answers,ground_truth = getBatchIndexedQAs(batch_qa,QA_words,v2i, nql=16, nqa=10, numOfChoices=2)
			input_video = getBatchVideoFeature(batch_qa, QA_words, hf, feature_shape)
			print(input_video)
			print(ground_truth)
			break
		break


if __name__=='__main__':
	main()