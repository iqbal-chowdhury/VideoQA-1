import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import numpy as np

import ModelUtil


def main():

	
	size_voc = 10

	video_feature_dims=100
	timesteps_v=10 # sequences length for video
	timesteps_q=11 # sequences length for question
	timesteps_a=12 # sequences length for anwser
	numberOfChoices = 2 # for input choices, one for correct, one for wrong answer

	word_embedding_size = 10
	sentence_embedding_size = 20
	visual_embedding_dims=25

	common_space_dim = 30

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
		loss = ModelUtil.getTripletLoss(T_v, T_q, T_a, y)

		
		# train module
		loss = tf.reduce_mean(loss)
		# acc_value = tf.metrics.accuracy(y, embeded_question)
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		train = optimizer.minimize(loss)

	# runtime environment 
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	with sess.as_default():
		for i in range(10000):
			batch_size = 64
			data_v = np.random.random((batch_size,timesteps_v,video_feature_dims))
			data_q = np.random.randint(0,10,size=(batch_size,timesteps_q),dtype='int32')
			data_a = np.random.randint(0,10,size=(batch_size, numberOfChoices, timesteps_a),dtype='int32')

			data_y = np.zeros((batch_size,numberOfChoices),dtype='float32')
			data_y[:,1]=1.0
			_, l = sess.run([train,loss],feed_dict={input_video:data_v, input_question:data_q, input_answer:data_a, y:data_y})
			print(l)

if __name__=='__main__':
	main()