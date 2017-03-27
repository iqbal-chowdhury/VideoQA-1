import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import numpy as np


def get_fans(shape):
	if len(shape) == 2:
		fan_in = shape[0]
		fan_out = shape[1]
	elif len(shape) == 4 or len(shape) == 5:
		receptive_field_size = np.prod(shape[2:])
		fan_in = shape[1] * receptive_field_size
		fan_out = shape[0] * receptive_field_size

	else:
		# No specific assumptions.
		fan_in = np.sqrt(np.prod(shape))
		fan_out = np.sqrt(np.prod(shape))
	return fan_in, fan_out


def uniform(shape, scale=0.05, name=None, seed=None): #tf.float32
	if seed is None:
		# ensure that randomness is conditioned by the Numpy RNG
		seed = np.random.randint(10e8)

	value = tf.random_uniform_initializer(
		-scale, scale, dtype=tf.float32, seed=seed)(shape)

	return tf.Variable(value)
    


def glorot_uniform(shape, name=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def orthogonal(shape, scale=1.1, name=None):
    """Orthogonal initializer.

    # References
        Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32, name=name)

def init_weight_variable(shape, init_method='glorot_uniform', name=None):
	# initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	if init_method == 'uniform':
		return uniform(shape, scale=0.05, name=name, seed=None)
	elif init_method == 'glorot_uniform':
		return glorot_uniform(shape, name=name)
	elif init_method == 'orthogonal':
		return orthogonal(shape, scale=1.1, name=name)
	else:
		raise ValueError('Invalid init_method: ' + init_method)
	
def init_bias_variable(shape,name=None):
	initial = tf.constant(0.1,shape=shape, name=name)
	return tf.Variable(initial)


def matmul_wx(x, w, b, output_dims):
	
	return tf.matmul(x, w)+tf.reshape(b,(1,output_dims))
	

def matmul_uh(u,h_tm1):
	return tf.matmul(h_tm1,u)



def get_init_state(x, output_dims):
	initial_state = tf.zeros_like(x)
	initial_state = tf.reduce_sum(initial_state,axis=[1,2])
	initial_state = tf.expand_dims(initial_state,dim=-1)
	initial_state = tf.tile(initial_state,[1,output_dims])
	return initial_state

'''
	function: getVideoEncoder
	parameters:

		x: batch_size, timesteps , dims
		output_dims: the output of the GRU dimensions
		num_class: number of class : ucf-101: 101

'''
# def getModel(x, timesteps, input_dims, output_dims, num_class):
def getVideoEncoder(x, output_dims, return_sequences=False):
	input_shape = x.get_shape().as_list()
	assert len(input_shape)==3 
	timesteps = input_shape[1]
	input_dims = input_shape[2]

	# get initial state
	initial_state = get_init_state(x, output_dims)

	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_r")
	W_z = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_z")
	W_h = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_h")

	U_r = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_r")
	U_z = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_z")
	U_h = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_h")

	b_r = init_bias_variable((output_dims,),name="b_r")
	b_z = init_bias_variable((output_dims,),name="b_z")
	b_h = init_bias_variable((output_dims,),name="b_h")


	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	x = tf.transpose(x, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims

	input_x = tf.TensorArray(
            dtype=x.dtype,
            size=timesteps,
            tensor_array_name='input_x')

	if hasattr(input_x, 'unstack'):
		input_x = input_x.unstack(x)
	else:
		input_x = input_x.unpack(x)	


	hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state')

	# if hasattr(hidden_state, 'unstack'):
	# 	hidden_state = hidden_state.unstack(hidden_state)
	# else:
	# 	hidden_state = hidden_state.unpack(hidden_state)


	def step(time, hidden_state, h_tm1):
		x_t = input_x.read(time) # batch_size * dim

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		h = (1-z)*hh + z*h_tm1

		hidden_state = hidden_state.write(time, h)

		return (time+1,hidden_state,h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state, initial_state),
            parallel_iterations=32,
            swap_memory=True)

	output = ret_out[1]
	last_output = ret_out[-1] 

	if hasattr(hidden_state, 'stack'):
		hidden_state = hidden_state.stack()

	axis = [1,0] + list(range(2,3))
	outputs = tf.transpose(hidden_state,perm=axis)

	# #linear classification
	# W_c = init_weight_variable((output_dims,num_class),name="W_c")
	# b_c = init_bias_variable((num_class,),name="b_r")
	# # call softmax function
	# # scores = tf.nn.softmax(tf.matmul(last_output,W_c)+b_c)

	# scores = tf.matmul(last_output,W_c)+b_c
	# # softmax implementation
	# # scores = tf.matmul(last_output,W_c)+b_c
	# # scores -= tf.argmax(scores,axis=-1,)
	# # scores /= tf.sum(scores,axis=-1)
	if return_sequences:
		return outputs
	else:
		return last_output

'''
	function: getEmbedding
		parameters:
			words: int, word index ; or a np.int32 list ## sample(null) * input_words_sequential
			size_voc: size of vocabulary
			embedding_size: the dimension after embedding
'''
def getEmbedding(words, size_voc, word_embedding_size):
	# words = list(words)
	# W_e = init_weight_variable((size_voc,word_embedding_size),init_method='uniform',name='W_e') # not share the variable

	W_e = tf.get_variable('W_e',(size_voc,word_embedding_size),initializer=tf.random_uniform_initializer(-0.05,0.05)) # share the embedding matrix
	embeded_words = tf.gather(W_e, words)
	mask =  tf.not_equal(words,0)
	return embeded_words, mask 



'''
	function: getQuestionEncoder
	parameters:
		embeded_words: sample*timestep*dim
		output_dims: the GRU hidden dim
		mask: bool type , samples * timestep
'''
def getQuestionEncoder(embeded_words, output_dims, mask, return_sequences=False):
	input_shape = embeded_words.get_shape().as_list()
	assert len(input_shape)==3 

	timesteps = input_shape[1]
	input_dims = input_shape[2]
	# get initial state
	initial_state = get_init_state(embeded_words, output_dims)


	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_r")
	W_z = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_z")
	W_h = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_h")

	U_r = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_r")
	U_z = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_z")
	U_h = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_h")

	b_r = init_bias_variable((output_dims,),name="b_q_r")
	b_z = init_bias_variable((output_dims,),name="b_q_z")
	b_h = init_bias_variable((output_dims,),name="b_q_h")


	# batch_size x timesteps x dim -> timesteps x batch_size x dim
	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
	embeded_words = tf.transpose(embeded_words, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



	input_embeded_words = tf.TensorArray(
            dtype=embeded_words.dtype,
            size=timesteps,
            tensor_array_name='input_embeded_words_q')


	if hasattr(input_embeded_words, 'unstack'):
		input_embeded_words = input_embeded_words.unstack(embeded_words)
	else:
		input_embeded_words = input_embeded_words.unpack(embeded_words)	


	# preprocess mask
	if len(mask.get_shape()) == len(input_shape)-1:
		mask = tf.expand_dims(mask,dim=-1)
	
	mask = tf.transpose(mask,perm=axis)

	input_mask = tf.TensorArray(
		dtype=mask.dtype,
		size=timesteps,
		tensor_array_name='input_mask_q'
		)

	if hasattr(input_mask, 'unstack'):
		input_mask = input_mask.unstack(mask)
	else:
		input_mask = input_mask.unpack(mask)


	hidden_state_q = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_q')

	# if hasattr(hidden_state, 'unstack'):
	# 	hidden_state = hidden_state.unstack(hidden_state)
	# else:
	# 	hidden_state = hidden_state.unpack(hidden_state)


	def step(time, hidden_state_q, h_tm1):
		x_t = input_embeded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		
		h = (1-z)*hh + z*h_tm1
		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

		h = tf.where(tiled_mask_t, h, h_tm1)
		
		hidden_state_q = hidden_state_q.write(time, h)

		return (time+1,hidden_state_q,h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state_q, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	hidden_state_q = ret_out[1]
	last_output = ret_out[-1] 
	
	if hasattr(hidden_state_q, 'stack'):
		outputs = hidden_state_q.stack()
		print('stack')
	else:
		outputs = hidden_state_q.pack()

	axis = [1,0] + list(range(2,3))
	outputs = tf.transpose(outputs,perm=axis)


	# #linear classification
	# W_c = init_weight_variable((output_dims,num_class),name="W_c_q")
	# b_c = init_bias_variable((num_class,),name="b_r_q")
	# # call softmax function
	# # scores = tf.nn.softmax(tf.matmul(last_output,W_c)+b_c)

	# scores = tf.matmul(last_output,W_c)+b_c
	if return_sequences:
		return outputs
	else:
		return last_output



'''
	function: getAnswerEmbedding
	parameters:
		words: int, word index ; or a np.int32 list ## sample(null) * numebrOfChoice * timesteps
		size_voc: size of vocabulary
		embedding_size: the dimension after embedding
'''
def getAnswerEmbedding(words, size_voc, word_embedding_size):
	assert len(words.get_shape().as_list())==3 #
	input_shape = words.get_shape().as_list()
	numberOfChoices = input_shape[1]
	timesteps = input_shape[2]

	mask =  tf.not_equal(words,0)

	words = tf.reshape(words, (-1,timesteps))
	W_e = tf.get_variable('W_e',(size_voc,word_embedding_size),initializer=tf.random_uniform_initializer(-0.05,0.05)) # share the embedding matrix
	embeded_words = tf.gather(W_e, words)
	

	embeded_words = tf.reshape(embeded_words,(-1,numberOfChoices,timesteps,word_embedding_size))
	
	return embeded_words, mask 


'''
	function: getAnswerEncoder
	parameters:
		embeded_words: samples * numberOfChoices * timesteps  * dim
		output_dim: output of GRU, the dimension of answering vector
		mask : bool type, mask the embeded_words
		num_class: number of classifier
'''
def getAnswerEncoder(embeded_words, output_dims, mask, return_sequences=False):
	input_shape = embeded_words.get_shape().as_list()
	assert len(input_shape)==4 


	numberOfChoices = input_shape[1]
	timesteps = input_shape[2]
	input_dims = input_shape[3]

	# get initial state
	embeded_words = tf.reshape(embeded_words,(-1,timesteps,input_dims))
	initial_state = get_init_state(embeded_words, output_dims)

	axis = [1,0,2]  
	embeded_words = tf.transpose(embeded_words, perm=axis) # permutate the 'embeded_words' --> timesteps x batch_size x numberOfChoices x dim
	# embeded_words = tf.reshape(embeded_words,(timesteps,-1,input_dims)) # reshape the 'embeded_words' --> timesteps x (batch x numberOfChoices) x dim
	
	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_r")
	W_z = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_z")
	W_h = init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_h")

	U_r = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_r")
	U_z = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_z")
	U_h = init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_h")

	b_r = init_bias_variable((output_dims,),name="b_a_r")
	b_z = init_bias_variable((output_dims,),name="b_a_z")
	b_h = init_bias_variable((output_dims,),name="b_a_h")



	input_embeded_words = tf.TensorArray(
            dtype=embeded_words.dtype,
            size=timesteps,
            tensor_array_name='input_embeded_words_a')


	if hasattr(input_embeded_words, 'unstack'):
		input_embeded_words = input_embeded_words.unstack(embeded_words)
	else:
		input_embeded_words = input_embeded_words.unpack(embeded_words)	


	# preprocess mask
	if len(mask.get_shape()) == len(input_shape)-1:
		mask = tf.expand_dims(mask,dim=-1)
	
	axis = [2,0,1,3]  
	mask = tf.transpose(mask,perm=axis)
	mask = tf.reshape(mask, (timesteps,-1,1))

	input_mask = tf.TensorArray(
		dtype=mask.dtype,
		size=timesteps,
		tensor_array_name='input_mask_q'
		)

	if hasattr(input_mask, 'unstack'):
		input_mask = input_mask.unstack(mask)
	else:
		input_mask = input_mask.unpack(mask)


	hidden_state_q = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_a')

	# if hasattr(hidden_state, 'unstack'):
	# 	hidden_state = hidden_state.unstack(hidden_state)
	# else:
	# 	hidden_state = hidden_state.unpack(hidden_state)


	def step(time, hidden_state_q, h_tm1):
		x_t = input_embeded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		
		h = (1-z)*hh + z*h_tm1
		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

		h = tf.where(tiled_mask_t, h, h_tm1)
		
		hidden_state_q = hidden_state_q.write(time, h)

		return (time+1,hidden_state_q,h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state_q, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	hidden_state_q = ret_out[1]
	last_output = ret_out[-1] 


	
	if hasattr(hidden_state_q, 'stack'):
		outputs = hidden_state_q.stack()
		print('stack')
	else:
		outputs = hidden_state_q.pack()

	outputs = tf.reshape(outputs,(timesteps,-1,numberOfChoices,output_dims))
	axis = [1,2,0]+list(range(3,4))
	outputs = tf.transpose(outputs,perm=axis)

	last_output = tf.reshape(last_output,(-1,numberOfChoices,output_dims))
	

	# outputs = tf.reshape(outputs, (-1,numberOfChoices,timesteps,output_dims)) # reshape the output ( batch_size, )


	# #linear classification
	# W_c = init_weight_variable((output_dims,num_class),name="W_c_a")
	# b_c = init_bias_variable((num_class,),name="b_r_a")
	# # call softmax function
	# # scores = tf.nn.softmax(tf.matmul(last_output,W_c)+b_c)

	# scores = tf.matmul(last_output,W_c)+b_c

	# # scores = tf.reshape(scores,(-1,numberOfChoices,num_class)) 
	# scores = tf.reshape(scores,(-1,numberOfChoices)) # num_class == 1 
	if return_sequences:
		return outputs
	else:
		return last_output

'''
	fucntion: getMultiModel
	parameters:
		visual_feature: batch_size * visual_encoded_dim
		question_feature: batch_size * question_encoded_dim
		answer_feature: batch_zize * numberOfChoices * answer_encoded_dim
		common_space_dim: embedding the visual,question,answer to the common space
	return: the embeded vectors(v,q,a)
'''

def getMultiModel(visual_feature, question_feature, answer_feature, common_space_dim):
	visual_shape = visual_feature.get_shape().as_list()
	question_shape = question_feature.get_shape().as_list()
	answer_shape = answer_feature.get_shape().as_list()

	# build the transformed matrix
	W_v = init_weight_variable((visual_shape[1],common_space_dim),init_method='glorot_uniform',name="W_v")
	W_q = init_weight_variable((question_shape[1],common_space_dim),init_method='glorot_uniform',name="W_q")
	W_a = init_weight_variable((answer_shape[2],common_space_dim),init_method='glorot_uniform',name="W_a")



	answer_feature = tf.reshape(answer_feature,(-1,answer_shape[2]))

	# encoder the features into common space
	T_v = tf.matmul(visual_feature,W_v)
	T_q = tf.matmul(question_feature,W_q)
	T_a = tf.matmul(answer_feature,W_a)

	T_a = tf.reshape(T_a,(-1,answer_shape[1],common_space_dim))

	return T_v,T_q,T_a
'''
	function: getRankingLoss
	parameters:
		answer_index: the ground truth index, one hot vector
	return:
		loss: tf.float32
'''
def getRankingLoss(T_v, T_q, T_a, answer_index=None,alpha = 0.2 ,isTest=False):
	
	# answer_index = tf.expand_dims(answer_index,dim=-1)
	# tf.tile(answer_index)
	# compute the loss
	
	T_v_shape = T_v.get_shape().as_list()
	T_q_shape = T_q.get_shape().as_list()
	T_a_shape = T_a.get_shape().as_list()

	numOfChoices = T_a_shape[1]
	common_space_dim = T_a_shape[2]

	assert T_q_shape == T_v_shape

	T_v = tf.nn.l2_normalize(T_v,1)
	T_q = tf.nn.l2_normalize(T_q,1)
	T_a = tf.nn.l2_normalize(T_a,2)

	T_p = tf.nn.l2_normalize(T_v+T_q,1)

	

	# answer_index = tf.tile(tf.expand_dims(answer_index,dim=-1),[1,1,T_q_shape[-1]]) # sample * numOfChoices * common_space_dim
	

	T_p = tf.tile(tf.expand_dims(T_p,dim=1),[1,numOfChoices,1])

	# T_p = tf.nn.l2_normalize(T_p*T_a,2)
	T_p = T_p*T_a
	T_p = tf.reduce_sum(T_p, reduction_indices=-1)

	scores = T_p

	if not isTest:
		assert answer_index is not None
		positive = tf.reduce_sum(T_p*answer_index, reduction_indices=1, keep_dims=True) # sample , get the positive score
		positive = tf.tile(positive,[1,numOfChoices])

		loss = (alpha - positive + T_p)*(1-answer_index)

		loss = tf.maximum(0.,loss)

		loss = tf.reduce_sum(loss,reduction_indices=-1)

		return loss,scores
	else:
		return scores

# def getRankingLoss_back(T_v, T_q, T_a, answer_index,alpha = 0.2 ):
	
# 	# answer_index = tf.expand_dims(answer_index,dim=-1)
# 	# tf.tile(answer_index)
# 	# compute the loss
	
# 	T_v_shape = T_v.get_shape().as_list()
# 	T_q_shape = T_q.get_shape().as_list()
# 	T_a_shape = T_a.get_shape().as_list()

# 	assert T_q_shape == T_v_shape

# 	T_a = tf.nn.l2_normalize(T_a,2)
# 	T_v = tf.nn.l2_normalize(T_v,1)
# 	T_q = tf.nn.l2_normalize(T_q,1)


# 	answer_index = tf.tile(tf.expand_dims(answer_index,dim=-1),[1,1,T_q_shape[-1]]) # sample * numOfChoices * common_space_dim


# 	right_answer =  tf.reduce_sum(T_a*answer_index,reduction_indices=1)
# 	error_answer =  tf.reduce_sum(T_a*(1.0-answer_index),reduction_indices=1)

	


# 	T_p = T_v+T_q 

# 	# --- normalization ---

# 	T_p = tf.nn.l2_normalize(T_p,1)
# 	right_answer = tf.nn.l2_normalize(right_answer,1)
# 	error_answer = tf.nn.l2_normalize(error_answer,1)


	
	
# 	gt = tf.reduce_sum(T_p*right_answer,reduction_indices=-1)
# 	ft = tf.reduce_sum(T_p*error_answer,reduction_indices=-1)
# 	loss = alpha - gt + ft
# 	# tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)
# 	loss = tf.maximum(0.,loss)
# 	return loss

'''
	function: getTripletLoss
	parameters:
		answer_index: the ground truth index, one hot vector
	return:
		loss: tf.float32
'''
def getTripletLoss(T_v, T_q, T_a, answer_index,alpha = 1 ):
	
	# answer_index = tf.expand_dims(answer_index,dim=-1)
	# tf.tile(answer_index)
	# compute the loss
	
	T_v_shape = T_v.get_shape().as_list()
	T_q_shape = T_q.get_shape().as_list()
	T_a_shape = T_a.get_shape().as_list()

	assert T_q_shape == T_v_shape

	

	answer_index = tf.tile(tf.expand_dims(answer_index,dim=-1),[1,1,T_q_shape[-1]]) # sample * numOfChoices * common_space_dim
	right_answer =  tf.reduce_sum(T_a*answer_index,reduction_indices=1)
	error_answer =  tf.reduce_sum(T_a*(1.0-answer_index),reduction_indices=1)

	
	T_v = tf.nn.l2_normalize(T_v,1)
	T_q = tf.nn.l2_normalize(T_q,1)

	T_p = T_v+T_q 

	# --- normalization ---

	T_p = tf.nn.l2_normalize(T_p,1)
	right_answer = tf.nn.l2_normalize(right_answer,1)
	error_answer = tf.nn.l2_normalize(error_answer,1)

	positive = tf.reduce_sum(tf.square(T_p-right_answer),reduction_indices=-1)
	negative = tf.reduce_sum(tf.square(T_p-error_answer),reduction_indices=-1)
	loss = alpha + positive - negative
	# tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)
	loss = tf.maximum(0.,loss)
	return loss


if __name__=='__main__':
	''' 
		for video encoding
	'''
	# timesteps=10
	# input_dims=100
	# output_dims=100
	# num_class=10

	# print('test..')
	# input_x = tf.placeholder(tf.float32, shape=(None, timesteps, input_dims),name='input_x')
	# y = tf.placeholder(tf.float32,shape=(None, num_class))

	# scores = getVideoEncoder(input_x, output_dims, num_class)
	# # train module
	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = scores))
	# acc_value = tf.metrics.accuracy(y, scores)
	# optimizer = tf.train.GradientDescentOptimizer(0.01)
	# train = optimizer.minimize(loss)

	# # runtime environment 
	# init = tf.global_variables_initializer()
	# sess = tf.Session()
	# sess.run(init)
	# with sess.as_default():
	# 	for i in range(1000):
	# 		data_x = np.random.random((100,timesteps,input_dims))
	# 		data_y = np.zeros((100,num_class))
	# 		data_y[:,1]=1
	# 		_, l = sess.run([train,loss],feed_dict={input_x:data_x,y:data_y})
	# 		print(l)

	# print('done..')
	'''
		for question encoding
	'''
	# print('test question encording ...')

	# size_voc = 10
	# timesteps=10
	# word_embedding_size = 10
	# words = [1,2,3,4,5]
	# num_class = 101
	# question_embedding_size = 100
	# with tf.variable_scope('share_embedding_matrix') as scope:
	# 	input_question = tf.placeholder(tf.int32, shape=(None,timesteps), name='input_question')
	# 	y = tf.placeholder(tf.float32,shape=(None, num_class))
	# 	# embeded_words1, mask1 = getEmbedding(input_question, size_voc, word_embedding_size)
	# 	# scope.reuse_variables() # notice this line for share the variable
	# 	embeded_words, mask = getEmbedding(input_question, size_voc, word_embedding_size)
	# 	embeded_question,outputs = getQuestionEncoder(embeded_words, question_embedding_size, mask, num_class)

	# 	# sess.run(init)
	# 	# train module
	# 	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = embeded_question))
	# 	acc_value = tf.metrics.accuracy(y, embeded_question)
	# 	optimizer = tf.train.GradientDescentOptimizer(0.01)
	# 	train = optimizer.minimize(loss)

	# # runtime environment 
	# init = tf.global_variables_initializer()
	# sess = tf.Session()
	# sess.run(init)
	# with sess.as_default():

	# 	for i in range(1):
	# 		data_x = np.random.randint(0,10,size=(2,timesteps),dtype='int32')
	# 		data_y = np.zeros((2,num_class))
	# 		data_y[:,1]=1
	# 		_, l, output_embed,mask_p,outpus_p = sess.run([train,loss,embeded_words,mask,outputs],feed_dict={input_question:data_x,y:data_y})
	# 		print(l)
	# 		print(data_x)
	# 		print(mask_p)

	'''
		for answering encoding
	'''
	# print('test answer encording ...')

	# size_voc = 10
	# timesteps=10
	# word_embedding_size = 10
	# words = [1,2,3,4,5]
	# num_class = 5
	# numberOfChoices = 5
	# question_embedding_size = 100

	# with tf.variable_scope('share_embedding_matrix') as scope:
	# 	input_question = tf.placeholder(tf.int32, shape=(None,numberOfChoices,timesteps), name='input_answer')
	# 	y = tf.placeholder(tf.float32,shape=(None, num_class))
	# 	# embeded_words1, mask1 = getEmbedding(input_question, size_voc, word_embedding_size)
	# 	# scope.reuse_variables() # notice this line for share the variable
	# 	embeded_words, mask = getAnswerEmbedding(input_question, size_voc, word_embedding_size)
	# 	embeded_question = getAnswerEncoder(embeded_words, question_embedding_size, mask, 1)
	# 	# sess.run(init)
	# 	# train module
	# 	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = embeded_question))
	# 	acc_value = tf.metrics.accuracy(y, embeded_question)
	# 	optimizer = tf.train.GradientDescentOptimizer(0.01)
	# 	train = optimizer.minimize(loss)

	# # runtime environment 
	# init = tf.global_variables_initializer()
	# sess = tf.Session()
	# sess.run(init)
	# with sess.as_default():

	# 	for i in range(1):
	# 		data_x = np.random.randint(0,10,size=(2,numberOfChoices, timesteps),dtype='int32')
	# 		data_y = np.zeros((2,num_class))
	# 		data_y[:,1]=1
	# 		_, l, output_embed,mask_p = sess.run([train,loss,embeded_words,mask],feed_dict={input_question:data_x,y:data_y})
	# 		print(l)
	# 		print(data_x)
	# 		print(mask_p)

	'''
		for training loss
	'''
	print('test answer encording ...')

	size_voc = 10
	timesteps=10
	word_embedding_size = 10
	words = [1,2,3,4,5]
	num_class = 5
	numberOfChoices = 2
	embedding_size = 100
	common_space_dim = 200
	with tf.variable_scope('share_embedding_matrix') as scope:
		visual_feature = tf.placeholder(tf.float32, shape=(None,embedding_size), name='visual_feature')
		question_feature = tf.placeholder(tf.float32, shape=(None,embedding_size), name='question_feature')
		answer_feature = tf.placeholder(tf.float32, shape=(None,numberOfChoices,embedding_size), name='answer_feature')
		y = tf.placeholder(tf.float32,shape=(None, numberOfChoices))
		# embeded_words1, mask1 = getEmbedding(input_question, size_voc, word_embedding_size)
		# scope.reuse_variables() # notice this line for share the variable
		T_v, T_q, T_a = getMultiModel(visual_feature, question_feature, answer_feature, common_space_dim)

		# loss, scores = getRankingLoss(T_v, T_q, T_a, y)
		loss = getTripletLoss(T_v, T_q, T_a, y)

		# embeded_words, mask = getAnswerEmbedding(input_question, size_voc, word_embedding_size)
		# embeded_question = getAnswerEncoder(embeded_words, question_embedding_size, mask, 1)
		# sess.run(init)
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
			data_v = np.random.random((batch_size,embedding_size))
			data_q = np.random.random((batch_size,embedding_size))
			data_a = np.random.random((batch_size,numberOfChoices, embedding_size))

			data_y = np.zeros((batch_size,numberOfChoices),dtype='float32')
			data_y[:,1]=1.0
			_, l = sess.run([train,loss],feed_dict={visual_feature:data_v, question_feature:data_q, answer_feature:data_a, y:data_y})
			print(l)



	


	



