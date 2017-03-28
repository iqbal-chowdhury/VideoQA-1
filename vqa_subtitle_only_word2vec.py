import numpy as np
import os
import h5py
import math

import MovieQA_benchmark as MovieQA
import DataUtil
import ModelUtil
import word2vec as w2v

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf


def build_model(input_stories, input_question, input_answer, word_embedding_size, sentence_embedding_size, v2i, w2v, 
			answer_index=None, lr=0.01,
			d_w2v=300, d_lproj=300,
			isTest=False):


	with tf.variable_scope('share_embedding_matrix') as scope:
		
		T_B, T_w2v, T_mask, pca_mat = ModelUtil.setWord2VecModelConfiguration(v2i,w2v,d_w2v,d_lproj)
		# encode question
		embeded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
		embeded_question = ModelUtil.getQuestionEncoder(embeded_question_words, sentence_embedding_size, mask_q)
		# embeded_question = ModelUtil.getAverageRepresentation(embeded_question_words)

		scope.reuse_variables()
		# encode stories
		embeded_stories_words, mask_s = ModelUtil.getEmbeddingWithWord2Vec(input_stories, T_w2v, T_mask)
		embeded_stories = ModelUtil.getMemoryNetworks(embeded_stories_words, embeded_question, mask_s)

		# encode answers
		embeded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
		embeded_answer = ModelUtil.getAnswerEncoder(embeded_answer_words, sentence_embedding_size, mask_a)
		# embeded_answer = ModelUtil.getAverageRepresentation(embeded_answer_words)

		# T_s, T_q, T_a = ModelUtil.getMultiModel(embeded_stories, embeded_question, embeded_answer, common_space_dim)
		

		if not isTest:
			# loss = ModelUtil.getTripletLoss(T_s, T_q, T_a, y)
			loss,scores = ModelUtil.getRankingLoss(embeded_stories, embeded_question, embeded_answer, answer_index=answer_index,isTest=isTest)


			
			# train module
			loss = tf.reduce_mean(loss)
			# acc_value = tf.metrics.accuracy(y, embeded_question)
			optimizer = tf.train.GradientDescentOptimizer(lr)
			train = optimizer.minimize(loss)
			return train,loss,scores
		else:
			scores = ModelUtil.getRankingLoss(embeded_stories, embeded_question, embeded_answer, answer_index=answer_index,isTest=isTest)
			return scores



def train_model(pretrained_model=None):
	task = 'video-based' # video-based or subtitle-based

	mqa = MovieQA.DataLoader()
	full_stories, full_video_QAs = mqa.get_story_qa_data('full', 'subtitle')
	full_stories = DataUtil.preprocess_stories(full_stories,max_words=10)

	w2v_mqa_model_filename = './model/movie_plots_1364.d-300.mc1.w2v'
	w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')


	# Create vocabulary
	v2i = DataUtil.create_vocabulary_word2vec(full_video_QAs, full_stories, word_thresh=2, w2v_vocab=w2v_model, v2i={'': 0, 'UNK':1})

	trained_stories, trained_video_QAs = mqa.get_story_qa_data('train', 'subtitle')
	trained_stories = DataUtil.preprocess_stories(trained_stories,max_words=10)
	trained_stories,max_sentences,max_words = DataUtil.data_in_matrix_form(trained_stories, v2i)

	print('full_stories... max setences:%d, max words:%d' %(max_sentences,max_words))


	val_stories, val_video_QAs = mqa.get_story_qa_data('val', 'subtitle')
	val_stories = DataUtil.preprocess_stories(val_stories,max_words=10)
	val_stories,_,_ = DataUtil.data_in_matrix_form(val_stories, v2i,max_sentences=max_sentences,max_words=max_words)

	'''
		model parameters
	'''
	# preprocess trained_stories

	size_voc = len(v2i)


	
	story_shape = (max_sentences,max_words)

	timesteps_q=16 # sequences length for question
	timesteps_a=10 # sequences length for anwser
	numberOfChoices = 5 # for input choices, one for correct, one for wrong answer

	word_embedding_size = 300
	sentence_embedding_size = 300
	


	print('building model ...')

	input_stories = tf.placeholder(tf.int32, shape=(None, max_sentences, max_words),name='input_stories')
	input_question = tf.placeholder(tf.int32, shape=(None,timesteps_q), name='input_question')
	input_answer = tf.placeholder(tf.int32, shape=(None,numberOfChoices,timesteps_a), name='input_answer')

	y = tf.placeholder(tf.float32,shape=(None, numberOfChoices))

	train,loss,scores = build_model(input_stories, input_question, input_answer, word_embedding_size, sentence_embedding_size, v2i, w2v_model, 
			answer_index=y, lr=0.01,
			d_w2v=300, d_lproj=300,
			isTest=False)


	'''
		configure && runtime environment
	'''
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.6
	# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	config.log_device_placement=False

	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	'''
		training parameters
	'''

	batch_size = 32
	total_train_qa = len(trained_video_QAs)
	total_val_qa = len(val_video_QAs)

	num_train_batch = int(round(total_train_qa*1.0/batch_size))
	num_val_batch = int(round(total_val_qa*1.0/batch_size))

	total_epoch = 50
	

	export_path = '/home/xyj/usr/local/saved_model/vqa_baseline/rankloss_subtitle_only_word2vec'
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
			# shuffle
			np.random.shuffle(trained_video_QAs)
			for batch_idx in xrange(num_train_batch):

				batch_qa = trained_video_QAs[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_train_qa)]


				data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa, v2i, nql=16, nqa=10, numOfChoices=numberOfChoices)
				data_s = DataUtil.getBatchIndexedStories(batch_qa,trained_stories,v2i,story_shape)
				_, l, s = sess.run([train,loss,scores],feed_dict={input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})

				num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
				Acc = num_correct*1.0/batch_size
				print('--Training--, Epoch: %d/%d, Batch: %d/%d, Batch_size: %d Loss: %.5f, Acc: %.5f' %(epoch+1,total_epoch,batch_idx+1,num_train_batch,batch_size,l,Acc))

			print('---------Validation---------')
			total_correct_num = 0
			for batch_idx in xrange(num_val_batch):

				batch_qa = val_video_QAs[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_val_qa)]


				data_q,data_a,data_y = DataUtil.getBatchIndexedQAs(batch_qa, v2i, nql=16, nqa=10, numOfChoices=numberOfChoices)
				data_s = DataUtil.getBatchIndexedStories(batch_qa,val_stories,v2i,story_shape)

				l, s = sess.run([loss,scores],feed_dict={input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})

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

	pretrained_model = '/home/xyj/usr/local/saved_model/vqa_baseline/rankloss_subtitle_only_word2vec/E15_A0.266598569969.ckpt'
	train_model(pretrained_model=pretrained_model)
	
	
	
	
	


	