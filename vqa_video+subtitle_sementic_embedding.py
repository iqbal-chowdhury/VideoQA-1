import numpy as np
import os
import h5py
import math

import MovieQA_benchmark as MovieQA
import DataUtil
import ModelUtil
import SemanticEmbeddingModelUtil as SEModelUtil
import word2vec as w2v

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from sklearn.decomposition import PCA
import cPickle as pickle
import time
import json
def build_model(input_video, input_stories, input_question, input_answer, 
			v2i,w2v_model,pca_mat=None,d_w2v=300,d_lproj=300,
			answer_index = None, lr=0.01, question_guided=False):


	with tf.variable_scope('share_embedding_matrix') as scope:
		

		T_B, T_w2v, T_mask, pca_mat = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj)
		# encode question
		embeded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
		embeded_question = ModelUtil.getAverageRepresentation(embeded_question_words,T_B,d_lproj)


		# encode video
		if question_guided:
			embeded_video = SEModelUtil.getVideoQuestionGuidedSemanticEmbedding(input_video, embeded_question,T_w2v, T_B, pca_mat=pca_mat)
		else:
			embeded_video = ModelUtil.getVideoSemanticEmbedding(input_video, T_w2v, T_B, pca_mat=pca_mat) # batch x timesteps x d_w2v

		# encode stories
		embeded_stories_words, mask_s = ModelUtil.getEmbeddingWithWord2Vec(input_stories, T_w2v, T_mask)
		embeded_stories = ModelUtil.getMemoryNetworks(embeded_stories_words, embeded_question, d_lproj, T_B=T_B)

		# encode answers
		embeded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
		embeded_answer = ModelUtil.getAverageRepresentation(embeded_answer_words,T_B,d_lproj)


		# get video loss
		video_loss,video_scores = ModelUtil.getClassifierLoss(embeded_video, embeded_question, embeded_answer, answer_index=answer_index)

		# get subtitle loss
		subtitle_loss,subtitle_scores = ModelUtil.getClassifierLoss(embeded_stories, embeded_question, embeded_answer, answer_index=answer_index)

		# late fussion
		loss = 1.0*(video_loss + subtitle_loss)/2

		scores = 1.0*(video_scores + subtitle_scores)/2
		
		# train module
		loss = tf.reduce_mean(loss)
		# acc_value = tf.metrics.accuracy(y, embeded_question)
		optimizer = tf.train.GradientDescentOptimizer(lr)
		train = optimizer.minimize(loss)
		return train,loss,scores
		
def linear_project_pca_initialization(hf,  feature_shape, d_w2v=300, output_path=None):

	print('--utilize PCA to initialize the embedding matrix of feature to d_w2v')
	samples = []
	for imdb_key in hf.keys():
		feature = hf[imdb_key][:]
		axis = [0,2,3,1]
		feature = np.transpose(feature, tuple(axis))
		feature = np.reshape(feature,(-1,feature_shape[1]))
		feature = np.random.permutation(feature)
		samples.extend(feature[:50])
	print('samples:',len(samples))

	pca = PCA(n_components=d_w2v, whiten=True)
	pca_mat = pca.fit_transform(np.asarray(samples).T)  # 1024 x 300

	pickle.dump(pca_mat,open(output_path,'w'))
	print('pca_amt dump to file:',output_path)
	return pca_mat


def exe_model(sess, data, batch_size, v2i, hf, feature_shape, stories, story_shape,
	loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=None, nql=25, nqa=32):
	if train is not None:
		np.random.shuffle(data)

	total_data = len(data)
	num_batch = int(round(total_data*1.0/batch_size))

	total_correct_num = 0
	total_loss = 0.0
	for batch_idx in xrange(num_batch):
		batch_qa = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
		
		data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa,v2i, nql=nql, nqa=nqa, numOfChoices=numberOfChoices)
		data_s = DataUtil.getBatchIndexedStories(batch_qa,stories,v2i,story_shape)
		data_v = DataUtil.getBatchVideoFeatureFromQid(batch_qa, hf, feature_shape)
		if train is not None:
			_, l, s = sess.run([train,loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})
		else:
			l, s = sess.run([loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})


		num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
		total_correct_num += num_correct
		total_loss += l
			# print('--Training--, Epoch: %d/%d, Batch: %d/%d, Batch_size: %d Loss: %.5f, Acc: %.5f' %(epoch+1,total_epoch,batch_idx+1,num_batch,batch_size,l,Acc))
	total_acc = total_correct_num*1.0/total_data
	total_loss = total_loss/num_batch
	return total_acc, total_loss





		
		

		



def train_model(hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=(16,1024,7,7),
		batch_size=8,total_epoch=100,
		lr=0.01,question_guided=False,pca_mat_init_file=None,pretrained_model=None):
	

	mqa = MovieQA.DataLoader()
	stories_for_create_dict, full_video_QAs = mqa.get_story_qa_data('full', 'subtitle')
	stories_for_create_dict = DataUtil.preprocess_stories(stories_for_create_dict,max_words=40)

	w2v_mqa_model_filename = './model/movie_plots_1364.d-300.mc1.w2v'
	w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')


	# Create vocabulary
	v2i = DataUtil.create_vocabulary_word2vec(full_video_QAs, stories_for_create_dict, 
		word_thresh=1, w2v_vocab=w2v_model, v2i={'': 0, 'UNK':1})

	# get 'video-based' QA task training set
	train_movies, trained_video_QAs = mqa.get_video_list('train', 'all_clips')  # key: 'train:<id>', value: list of related clips
	val_movies, val_video_QAs = mqa.get_video_list('val', 'all_clips')


	train_stories, val_stories = DataUtil.split_stories(stories_for_create_dict,train_movies,val_movies)

	train_stories,max_sentences,max_words = DataUtil.data_in_matrix_form(train_stories, v2i)

	val_stories,_,_ = DataUtil.data_in_matrix_form(val_stories, v2i)

	story_shape = (max_sentences,max_words)

	size_voc = len(v2i)



	print('building model ...')
	
	if os.path.exists(pca_mat_init_file):
		pca_mat = pickle.load(open(pca_mat_init_file,'r'))
	else:
		pca_mat = linear_project_pca_initialization(hf, feature_shape, d_w2v=300, output_path=pca_mat_init_file)

	print('pca_mat.shape:',pca_mat.shape)

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_stories = tf.placeholder(tf.int32, shape=(None, max_sentences, max_words),name='input_stories')
	input_question = tf.placeholder(tf.int32, shape=(None,nql), name='input_question')
	input_answer = tf.placeholder(tf.int32, shape=(None,numberOfChoices,nqa), name='input_answer')

	y = tf.placeholder(tf.float32,shape=(None, numberOfChoices))

	train,loss,scores = build_model(input_video, input_stories, input_question, input_answer, v2i,w2v_model,
			pca_mat=pca_mat,
			d_w2v=300,d_lproj=300,
			answer_index=y,  lr=lr,
			question_guided=question_guided)

	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.35
	# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	config.log_device_placement=False

	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	'''
		training parameters
	'''

	with open('train_split.json') as fid:
		trdev = json.load(fid)


	def getTrainDevSplit(trained_video_QAs,trdev):
		train_data = []
		dev_data = []
		for k, qa in enumerate(trained_video_QAs):

			if qa.imdb_key in trdev['train']:
				train_data.append(qa)
			else:
				dev_data.append(qa)
		return train_data,dev_data

  	train_data,dev_data = getTrainDevSplit(trained_video_QAs,trdev)



	



	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
		if pretrained_model is not None:
			saver.restore(sess, pretrained_model)
			print('restore pre trained file:' + pretrained_model)
		for epoch in xrange(total_epoch):
			
			# # shuffle
			print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
			# train phase
			tic = time.time()
			total_acc, total_loss = exe_model(sess, train_data, batch_size, v2i, hf, feature_shape, train_stories, story_shape,
				loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=train, nql=25, nqa=32)
			print('    --Train--, Loss: %.5f, Acc: %.5f.......Time:%.3f' %(total_loss,total_acc,time.time()-tic))

			# dev phase
			tic = time.time()
			total_acc, total_loss = exe_model(sess, dev_data, batch_size, v2i, hf, feature_shape, train_stories, story_shape,
				loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=None, nql=25, nqa=32)
			print('    --Train-val--, Loss: %.5f, Acc: %.5f.......Time:%.3f' %(total_loss,total_acc,time.time()-tic))
			# eval phase
			# if total_acc > max_acc:
			# 	max_acc = total_acc
			tic = time.time()
			total_acc, total_loss = exe_model(sess, val_video_QAs, batch_size, v2i, hf, feature_shape, val_stories, story_shape,
				loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=None, nql=25, nqa=32)
			print('    --Val--,  Loss: %.5f, Acc: %.5f.......Time:%.3f' %(total_loss,total_acc,time.time()-tic))



			#save model
			if question_guided:
				export_path = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_semantic_qguided'+'_'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])
			else:
				export_path = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_semantic'+'_'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])
			if not os.path.exists(export_path):
				os.makedirs(export_path)
				print('mkdir %s' %export_path)
			save_path = saver.save(sess, export_path+'/'+'E'+str(epoch+1)+'_A'+str(total_acc)+'.ckpt')
			print("Model saved in file: %s" % save_path)
		


if __name__ == '__main__':
	# 'video+subtitle task'

	
	
	nql=25 # sequences length for question
	nqa=32 # sequences length for anwser
	numberOfChoices = 5 # for input choices, one for correct, one for wrong answer

	lr = 0.01
	'''
	---------------------------------
	'''
	video_feature_dims=1024
	timesteps_v=16 # sequences length for video
	hight = 7
	width = 7
	feature_shape = (timesteps_v,video_feature_dims,hight,width)

	f_type = 'GoogLeNet'
	feature_path = '/home/xyj/usr/local/data/movieqa/movie_google_'+str(timesteps_v)+'f.h5'
	pca_mat_init_file = '/home/xyj/usr/local/code/VideoQA/model/google_pca_mat.pkl'

	'''
	---------------------------------
	'''
	# video_feature_dims=256
	# timesteps_v=16 # sequences length for video
	# hight = 6
	# width = 6
	# feature_shape = (timesteps_v,video_feature_dims,hight,width)

	# f_type = 'HybridCNN'
	# feature_path = '/home/xyj/usr/local/data/movieqa/movie_hybridcnn_'+str(timesteps_v)+'f.h5'
	# pca_mat_init_file = '/home/xyj/usr/local/code/VideoQA/model/hybridcnn_pca_mat.pkl'

	'''
	---------------------------------
	'''
	hf = h5py.File(feature_path,'r')


	# pretrained_model = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_semantic_HybridCNN/lr0.01_f16/E10_A0.36230248307.ckpt'
	pretrained_model = None
	train_model(hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=feature_shape,lr=lr,
		batch_size=8,total_epoch=100,
		pretrained_model=pretrained_model,pca_mat_init_file=pca_mat_init_file,question_guided=False)
	

	
	
	
	


	