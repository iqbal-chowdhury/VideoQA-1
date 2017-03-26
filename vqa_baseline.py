import numpy as np
import os
import h5py
import math

import MovieQA_benchmark as MovieQA
import DataUtil
import ModelUtil


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf


def train():
	task = 'video-based' # video-based or subtitle-based

	mqa = MovieQA.DataLoader()

	# get 'subtitile-based' QA task dataset
	stories, subtitle_QAs = mqa.get_story_qa_data('train', 'subtitle')

	# Create vocabulary
	QA_words, v2i = DataUtil.create_vocabulary(subtitle_QAs, stories, word_thresh=2, v2i={'': 0, 'UNK':1})

	# get 'video-based' QA task training set
	_, trained_video_QAs = mqa.get_video_list('train', 'qa_clips')  # key: 'train:<id>', value: list of related clips
	# vl_qa, _ = mqa.get_video_list('train', 'all_clips') # key:moive vid, value:list of related movid all_clips

	_, val_video_QAs = mqa.get_video_list('val', 'qa_clips')


	'''
		model parameters
	'''
	size_voc = len(v2i)

	video_feature_dims=2048
	timesteps_v=16 # sequences length for video
	feature_shape = (timesteps_v,video_feature_dims)

	timesteps_q=16 # sequences length for question
	timesteps_a=10 # sequences length for anwser
	numberOfChoices = 5 # for input choices, one for correct, one for wrong answer

	word_embedding_size = 300
	sentence_embedding_size = 512
	visual_embedding_dims=512

	common_space_dim = 512
	

	print('test..')
	with tf.variable_scope('share_embedding_matrix') as scope:
		input_video = tf.placeholder(tf.float32, shape=(None, timesteps_v, video_feature_dims),name='input_video')
		input_question = tf.placeholder(tf.int32, shape=(None,timesteps_q), name='input_question')
		input_answer = tf.placeholder(tf.int32, shape=(None,numberOfChoices,timesteps_a), name='input_answer')

		y = tf.placeholder(tf.float32,shape=(None, numberOfChoices))

		embeded_video = ModelUtil.getVideoEncoder(input_video, visual_embedding_dims)

		embeded_question_words, mask_q = ModelUtil.getEmbedding(input_question, size_voc, word_embedding_size)
		embeded_question = ModelUtil.getQuestionEncoder(embeded_question_words, sentence_embedding_size, mask_q)

		scope.reuse_variables()
		embeded_answer_words, mask_a = ModelUtil.getAnswerEmbedding(input_answer, size_voc, word_embedding_size)
		embeded_answer = ModelUtil.getAnswerEncoder(embeded_answer_words, sentence_embedding_size, mask_a)


		T_v, T_q, T_a = ModelUtil.getMultiModel(embeded_video, embeded_question, embeded_answer, common_space_dim)
		# loss = ModelUtil.getTripletLoss(T_v, T_q, T_a, y)
		loss,scores = ModelUtil.getRankingLoss(T_v, T_q, T_a, y)


		
		# train module
		loss = tf.reduce_mean(loss)
		# acc_value = tf.metrics.accuracy(y, embeded_question)
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		train = optimizer.minimize(loss)


	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
	# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	config.log_device_placement=False

	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	'''
		training parameters
	'''

	batch_size = 64
	total_train_qa = len(trained_video_QAs)
	total_val_qa = len(val_video_QAs)

	num_train_batch = int(round(total_train_qa*1.0/batch_size))
	num_val_batch = int(round(total_val_qa*1.0/batch_size))

	total_epoch = 100
	f_type = 'res'
	if f_type=='res':
		feature_path = '/home/wb/res_movie_feature.hdf5'
	else:
		feature_path = '/home/wb/movie_feature.hdf5'
	# /home/wb/res_movie_feature.hdf5
	
	
	hf = h5py.File(feature_path,'r')

	export_path = '/home/xyj/usr/local/saved_model/vqa_baseline/rankloss'+'_'+f_type
	if not os.path.exists(export_path):
		os.makedirs(export_path)
		print('mkdir %s' %export_path)

	print('total training samples: %d' %total_train_qa)

	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
		# saver.restore(sess, export_path+'/iter113_L1.12049588864_A0.751643375464.ckpt')
		for epoch in xrange(total_epoch):
			# # shuffle
			np.random.shuffle(trained_video_QAs)
			for batch_idx in xrange(num_train_batch):

				batch_qa = trained_video_QAs[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_qa)]


				data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa,QA_words,v2i, nql=16, nqa=10, numOfChoices=numberOfChoices)
				data_v = DataUtil.getBatchVideoFeature(batch_qa, QA_words, hf, feature_shape)
				_, l, s = sess.run([train,loss,scores],feed_dict={input_video:data_v, input_question:data_q, input_answer:data_a, y:data_y})

				num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
				Acc = num_correct*1.0/batch_size
				print('--Training--, Epoch: %d/%d, Batch: %d/%d, Batch_size: %d Loss: %.5f, Acc: %.5f' %(epoch+1,total_epoch,batch_idx+1,num_train_batch,batch_size,l,Acc))

			print('---------Validation---------')
			total_correct_num = 0
			for batch_idx in xrange(num_val_batch):

				batch_qa = val_video_QAs[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_val_qa)]


				data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa,QA_words,v2i, nql=16, nqa=10, numOfChoices=numberOfChoices)
				data_v = DataUtil.getBatchVideoFeature(batch_qa, QA_words, hf, feature_shape)

				l, s = sess.run([loss,scores],feed_dict={input_video:data_v, input_question:data_q, input_answer:data_a, y:data_y})

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
	train()