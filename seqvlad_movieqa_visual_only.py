import numpy as np
import os
import h5py
import math

import MovieQA_benchmark as MovieQA
import DataUtil
import ModelUtil
import HierarchicalSemanticEmbeddingModelUtil as HSEModelUtil
import word2vec as w2v

from seqvlad_model import SeqVladModel

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from sklearn.decomposition import PCA
import cPickle as pickle
import time
import json
from collections import namedtuple

def build_model(input_video, input_stories, input_question, input_answer, 
			v2i,w2v_model,pca_mat=None,d_w2v=300,d_lproj=300,
			answer_index = None, lr=0.01, isTest=False, question_guided=False):


	with tf.variable_scope('video_subtitle_hierarchical_frame') as scope:
		
		T_B, T_w2v, T_mask, pca_mat_ = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj)
		# encode question
		embedded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
		embedded_question = HSEModelUtil.getAverageRepresentation(embedded_question_words,T_B,d_lproj)

		# encode stories
		# embedded_stories_words, mask_s = ModelUtil.getEmbeddingWithWord2Vec(input_stories, T_w2v, T_mask)
		# embedded_stories = ModelUtil.getMemoryNetworks(embedded_stories_words, embedded_question, d_lproj, T_B=T_B, return_sequences=True)

		# encode video
		# embedded_video = HSEModelUtil.getVideoDualSemanticEmbedding(input_video, T_w2v, embedded_stories, T_B, pca_mat=pca_mat) # batch x timesteps x d_w2v
		# print('pca_mat:',pca_mat)
		# embedded_video = HSEModelUtil.getVideoSemanticEmbedding(input_video, T_w2v, T_B, pca_mat=pca_mat)

		seqvlad = SeqVladModel.SeqVladWithReduAttentionModel(input_video,
								d_w2v=d_w2v,
								reduction_dim=512,
								centers_num=64, 
								filter_size=3)
		vlad_feature = seqvlad.build_model()

		# encode answers
		embedded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
		embedded_answer = HSEModelUtil.getAverageRepresentation(embedded_answer_words,T_B,d_lproj)

		video_loss,video_scores = ModelUtil.getClassifierLoss(vlad_feature, embedded_question, embedded_answer, answer_index=answer_index)

		if isTest:
			# get video loss
			video_scores = ModelUtil.getClassifierLoss(vlad_feature, embedded_question, embedded_answer, answer_index=answer_index, isTest=isTest)


			return video_scores
		else:
			# train module
			loss = tf.reduce_mean(video_loss)
			optimizer = tf.train.GradientDescentOptimizer(lr)
			train = optimizer.minimize(loss)
			return train,loss,video_scores
		
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
		
		data_q,data_a,data_y = DataUtil.getBatchIndexedQAs_return(batch_qa,v2i, nql=nql, nqa=nqa, numOfChoices=numberOfChoices)
		data_s = DataUtil.getBatchIndexedStories(batch_qa,stories,v2i,story_shape)
		data_v = DataUtil.getBatchVideoFeatureFromQid(batch_qa, hf, feature_shape)
		if train is not None:
			_, l, s = sess.run([train,loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})
		else:
			l, s = sess.run([loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})


		num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
		total_correct_num += num_correct
		total_loss += l
		# print('--Training--,  Batch: %d/%d, Batch_size: %d Loss: %.5f' %(batch_idx+1,num_batch,batch_size,l))
	total_acc = total_correct_num*1.0/total_data
	total_loss = total_loss/num_batch
	return total_acc, total_loss




def test_model(sess, data, batch_size, v2i, hf, feature_shape, stories, story_shape,
	scores, input_video, input_question, input_stories, input_answer, numberOfChoices=5, nql=25, nqa=32, output_file=None):
	
	with open(output_file,'w') as wf:
		total_data = len(data)
		num_batch = int(math.ceil(total_data*1.0/batch_size))

		for batch_idx in xrange(num_batch):
			batch_qa = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
			
			data_q,data_a = DataUtil.getTestBatchIndexedQAs_return(batch_qa,v2i, nql=nql, nqa=nqa, numOfChoices=numberOfChoices)
			data_s = DataUtil.getBatchIndexedStories(batch_qa,stories,v2i,story_shape)
			data_v = DataUtil.getBatchVideoFeatureFromQid(batch_qa, hf, feature_shape)
			
			s = sess.run([scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a})

			for idx,qa in enumerate(batch_qa):
				wf.write('%s %d\n' %(qa.qid,np.argmax(s[0][idx])))
		
		

		



def train_model(train_stories,val_stories,v2i,trained_video_QAs,val_video_QAs,hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=(16,1024,7,7),
		batch_size=8,total_epoch=100,
		lr=0.01,pretrained_model=False,isTest=None,pca_mat_init_file=None,output_file=None):
	

	


	w2v_mqa_model_filename = './model/movie_plots_1364.d-300.mc1.w2v'
	w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')


	'''
		model parameters
	'''
	size_voc = len(v2i)
	max_sentences = 3660
	max_words = 40
	
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

	if isTest:
		scores = build_model(input_video, input_stories, input_question, input_answer, v2i,w2v_model,
			pca_mat=pca_mat,
			d_w2v=300,d_lproj=300,
			answer_index=y,  lr=lr, isTest=isTest)
	else:

		train,loss,scores = build_model(input_video, input_stories, input_question, input_answer, v2i,w2v_model,
				pca_mat=pca_mat,
				d_w2v=300,d_lproj=300,
				answer_index=y,  lr=lr)

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

	if isTest:
  		train_data = trained_video_QAs
	else:
		train_data,dev_data = getTrainDevSplit(trained_video_QAs,trdev)



	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
		if pretrained_model is not None:
			saver.restore(sess, pretrained_model)
			print('restore pre trained file:' + pretrained_model)


		if isTest:
			mqa = MovieQA.DataLoader()
			full_movies, test_qa = mqa.get_video_list('test', 'all_clips')
			test_qa_list = [qa.qid for qa in test_qa]
			used_test_data = []
			for qa in train_data:
				if qa.qid in test_qa_list:
					used_test_data.append(qa)

			print('total test data:', len(used_test_data))

			test_model(sess, used_test_data, 1, v2i, hf, feature_shape, train_stories, story_shape,
					scores, input_video, input_question, input_stories, input_answer, numberOfChoices=5, nql=25, nqa=32, output_file=output_file)
		else:

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
				export_path = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_hierarchical_frame'+'/'+f_type+'_b'+str(batch_size)+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])
				if not os.path.exists(export_path):
					os.makedirs(export_path)
					print('mkdir %s' %export_path)
				save_path = saver.save(sess, export_path+'/'+'E'+str(epoch+1)+'_A'+str(total_acc)+'.ckpt')
				print("Model saved in file: %s" % save_path)


def trans(all):

    qa_list = []
    for dicts in all:

        qa_list.append(
            QAInfo(dicts['qid'], dicts['questions'], dicts['answers'] , dicts['ground_truth'],
                   dicts['imdb_key'], dicts['video_clips']))
    return qa_list		

def transTest(all):

    qa_list = []
    for dicts in all:

        qa_list.append(
            QAInfo(dicts['qid'], dicts['questions'], dicts['answers'],
                   dicts['imdb_key'], dicts['video_clips']))
    return qa_list

if __name__ == '__main__':
	# 'video+subtitle task'

	
	
	nql=25 # sequences length for question
	nqa=32 # sequences length for anwser
	numberOfChoices = 5 # for input choices, one for correct, one for wrong answer
	QAInfo = namedtuple('QAInfo','qid question answers correct_index imdb_key video_clips')
	

	v2i = pickle.load(open("/home/wb/movieQA_v2i.pkl","rb"))
	qa_train = trans(pickle.load(open("/mnt/data3/yuan/processed_data/processed_video_qa/process_train.pkl","rb")))
	qa_val = trans(pickle.load(open("/mnt/data3/yuan/processed_data/processed_video_qa/process_val.pkl","rb")))
	train_stories = pickle.load(open("/mnt/data3/yuan/processed_data/processed_video_qa/train_stories.pkl","rb"))
	val_stories = pickle.load(open("/mnt/data3/yuan/processed_data/processed_video_qa/val_stories.pkl","rb"))

	lr = 0.01



	'''
	---------------------------------
	224x224 vgg all clips feature
	'''

	video_feature_dims=1024
	timesteps_v=32 # sequences length for video
	hight = 7
	width = 7
	feature_shape = (timesteps_v,video_feature_dims,hight,width)

	f_type = '224x224_GoogLeNet_all_clips'
	feature_path = '/mnt/data3/yuan/processed_data/224x224_movie_all_clips_google_'+str(timesteps_v)+'f.h5'
	pca_mat_init_file = '/home/xyj/usr/local/code/VideoQA/model/224x224_google_all_clip_pca_mat.pkl'



	hf = h5py.File(feature_path,'r')


	pretrained_model = '/home/xyj/usr/local/saved_model/vqa_baseline/video+subtitle_classifier_hierarchical_frame/224x224_GoogLeNet_all_clips_b8/lr0.01_f32/E12_A0.399548532731.ckpt'
	# pretrained_model = None


	isTest = True
	if isTest:

		QAInfo = namedtuple('QAInfo','qid question answers imdb_key video_clips')
 		qa_test = transTest(pickle.load(open("/mnt/data3/yuan/processed_data/processed_all_qa/process_test_all.pkl","rb")))
		test_stories = pickle.load(open("/mnt/data3/yuan/processed_data/processed_all_qa/test_stories_all.pkl","rb"))
 		output_file='/home/xyj/usr/local/predict_result/seqvlad_movie_qa/seqvlad_result.txt'
		train_model(test_stories,val_stories,v2i,qa_test,qa_val,hf,f_type,nql=25,nqa=32,numberOfChoices=5,
			feature_shape=feature_shape,lr=lr,
			batch_size=8,total_epoch=20,
			pretrained_model=pretrained_model,pca_mat_init_file=pca_mat_init_file,isTest=isTest,output_file=output_file)
	else:
		train_model(train_stories,val_stories,v2i,qa_train,qa_val,hf,f_type,nql=25,nqa=32,numberOfChoices=5,
			feature_shape=feature_shape,lr=lr,
			batch_size=8,total_epoch=20,
			pretrained_model=pretrained_model,pca_mat_init_file=pca_mat_init_file)
	

	
	
	
	


	
