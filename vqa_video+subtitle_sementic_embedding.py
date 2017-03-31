import numpy as np
import os
import h5py
import math

import MovieQA_benchmark as MovieQA
import DataUtil
import ModelUtil
import word2vec as w2v

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from sklearn.decomposition import PCA
import cPickle as pickle
import time

def build_model(input_video, input_stories, input_question, input_answer, 
			v2i,w2v_model,pca_mat=None,d_w2v=300,d_lproj=300,
			answer_index = None, lr=0.01):


	with tf.variable_scope('share_embedding_matrix') as scope:
		

		T_B, T_w2v, T_mask, pca_mat = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj)
		# encode question
		embeded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
		embeded_question = ModelUtil.getAverageRepresentation(embeded_question_words,T_B,d_lproj)


		# encode video
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
	imdb_keys = hf.keys()
	imdb_keys = np.random.permutation(imdb_keys)[:256]
	samples = []
	for imdb_key in imdb_keys:
		clips = hf[imdb_key].keys()
		clips = np.random.permutation(clips)
		for idx,clip in enumerate(clips):
			if idx<4:
				dataset = imdb_key+'/'+clip
				feature = hf[dataset][:]
				axis = [0,2,3,1]
				feature = np.transpose(feature, tuple(axis))
				feature = np.reshape(feature,(-1,feature_shape[1]))
				feature = np.random.permutation(feature)
				samples.extend(feature[:500])
	print('samples:',len(samples))

	pca = PCA(n_components=d_w2v, whiten=True)
	pca_mat = pca.fit_transform(np.asarray(samples).T)  # 1024 x 300

	pickle.dump(pca_mat,open(output_path,'w'))
	print('pca_amt dump to file:',output_path)
	return pca_mat

def train_model(hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=(16,1024,7,7),
		batch_size=8,total_epoch=100,
		lr=0.01,pretrained_model=None):
	

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
	init_file = '/home/xyj/usr/local/code/VideoQA/model/pca_mat.pkl'
	if init_file is None:
		output_path = '/home/xyj/usr/local/code/VideoQA/model/pca_mat.pkl'
		pca_mat = linear_project_pca_initialization(hf, feature_shape, d_w2v=300, output_path=output_path)
	else:
		pca_mat = pickle.load(open(init_file,'r'))

	print('pca_mat.shape:',pca_mat.shape)

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_stories = tf.placeholder(tf.int32, shape=(None, max_sentences, max_words),name='input_stories')
	input_question = tf.placeholder(tf.int32, shape=(None,nql), name='input_question')
	input_answer = tf.placeholder(tf.int32, shape=(None,numberOfChoices,nqa), name='input_answer')

	y = tf.placeholder(tf.float32,shape=(None, numberOfChoices))

	train,loss,scores = build_model(input_video, input_stories, input_question, input_answer, v2i,w2v_model,
			pca_mat=pca_mat,
			d_w2v=300,d_lproj=300,
			answer_index = y,  lr=lr)

	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
	# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	config.log_device_placement=False

	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	'''
		training parameters
	'''

	
	total_train_qa = len(trained_video_QAs)
	total_val_qa = len(val_video_QAs)

	num_train_batch = int(round(total_train_qa*1.0/batch_size))
	num_val_batch = int(round(total_val_qa*1.0/batch_size))


	export_path = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_semantic'+'_'+f_type+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])
	if not os.path.exists(export_path):
		os.makedirs(export_path)
		print('mkdir %s' %export_path)

	print('total training samples: %d' %total_train_qa)


	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
		if pretrained_model is not None:
			saver.restore(sess, pretrained_model)
			print('restore pre trained file:' + pretrained_model)
		for epoch in xrange(total_epoch):
			np.random.shuffle(trained_video_QAs) # shuffle
			for batch_idx in xrange(num_train_batch):

				batch_qa = trained_video_QAs[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_qa)]

				
				data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa,v2i, nql=nql, nqa=nqa, numOfChoices=numberOfChoices)
				data_s = DataUtil.getBatchIndexedStories(batch_qa,train_stories,v2i,story_shape)
				data_v = DataUtil.getBatchVideoFeatureFromQid(batch_qa, hf, feature_shape)

				_, l, s = sess.run([train,loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})

				num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
				Acc = num_correct*1.0/batch_size
				print('--Training--, Epoch: %d/%d, Batch: %d/%d, Batch_size: %d Loss: %.5f, Acc: %.5f' %(epoch+1,total_epoch,batch_idx+1,num_train_batch,batch_size,l,Acc))

			print('---------Validation---------')
			total_correct_num = 0
			for batch_idx in xrange(num_val_batch):

				batch_qa = val_video_QAs[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_val_qa)]


				data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa,v2i, nql=nql, nqa=nqa, numOfChoices=numberOfChoices)
				data_s = DataUtil.getBatchIndexedStories(batch_qa,val_stories,v2i,story_shape)
				data_v = DataUtil.getBatchVideoFeatureFromQid(batch_qa, hf, feature_shape)

				l, s = sess.run([loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})

				num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
				Acc = num_correct*1.0/batch_size
				total_correct_num += num_correct
				print('--Valid--, Epoch: %d/%d, Batch: %d/%d, Batch_size: %d Loss: %.5f, Acc: %.5f' %(epoch+1,total_epoch,batch_idx+1,num_val_batch,batch_size,l,Acc))
			total_correct_num = total_correct_num*1.0/total_val_qa
			print('--Valid--, val acc: %.5f' %(total_correct_num))

			#save model
			save_path = saver.save(sess, export_path+'/'+'E'+str(epoch+1)+'_A'+str(total_correct_num)+'.ckpt')
			print("Model saved in file: %s" % save_path)
		


if __name__ == '__main__':
	# 'video+subtitle task'

	
	video_feature_dims=1024
	timesteps_v=32 # sequences length for video
	hight = 7
	width = 7
	feature_shape = (timesteps_v,video_feature_dims,hight,width)

	nql=25 # sequences length for question
	nqa=32 # sequences length for anwser
	numberOfChoices = 5 # for input choices, one for correct, one for wrong answer

	lr = 0.01


	f_type = 'GoogLeNet'
	feature_path = '/home/xyj/usr/local/data/movieqa/movie_google_'+str(timesteps_v)+'f.h5'
	hf = h5py.File(feature_path,'r')


	# pretrained_model = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_semantic_GoogLeNet/E29_A0.379232505643_lr0.01.ckpt'

	train_model(hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=feature_shape,lr=lr,
		batch_size=8,total_epoch=100,
		pretrained_model=None)
	

	
	
	
	


	