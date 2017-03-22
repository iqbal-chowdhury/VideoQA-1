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
	x: batch_size, timesteps , dims
	num_class: number of class : ucf-101: 101
'''
# def getModel(x, timesteps, input_dims, output_dims, num_class):
def getVideoEncoder(x, output_dims, num_class):
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

		hidden_state.write(time, h)

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
	output = tf.transpose(hidden_state,perm=axis)

	#linear classification
	W_c = init_weight_variable((output_dims,num_class),name="W_c")
	b_c = init_bias_variable((num_class,),name="b_r")
	# call softmax function
	# scores = tf.nn.softmax(tf.matmul(last_output,W_c)+b_c)

	scores = tf.matmul(last_output,W_c)+b_c
	# softmax implementation
	# scores = tf.matmul(last_output,W_c)+b_c
	# scores -= tf.argmax(scores,axis=-1,)
	# scores /= tf.sum(scores,axis=-1)

	return scores

'''
	function: getEmbedding
		parameters:
			words: int, word index ; or a np.int32 list ## sample(null) * input_words_sequential
			size_voc: size of vocabulary
			embedding_size: the dimension after embedding
'''
def getEmbedding(words, size_voc, word_embedding_size):
	# words = list(words)
	W_e = init_weight_variable((size_voc,word_embedding_size),init_method='uniform',name='W_e')
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
def getQuestionEncoder(embeded_words, output_dims, mask, num_class):
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


	hidden_state = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_q')

	# if hasattr(hidden_state, 'unstack'):
	# 	hidden_state = hidden_state.unstack(hidden_state)
	# else:
	# 	hidden_state = hidden_state.unpack(hidden_state)


	def step(time, hidden_state, h_tm1):
		x_t = input_embeded_words.read(time) # batch_size * dim
		mask_t = input_mask.read(time)

		preprocess_x_r = matmul_wx(x_t, W_r, b_r, output_dims)
		preprocess_x_z = matmul_wx(x_t, W_z, b_z, output_dims)
		preprocess_x_h = matmul_wx(x_t, W_h, b_h, output_dims)

		r = tf.nn.sigmoid(preprocess_x_r+ matmul_uh(U_r,h_tm1))
		z = tf.nn.sigmoid(preprocess_x_z+ matmul_uh(U_z,h_tm1))
		hh = tf.nn.tanh(preprocess_x_h+ matmul_uh(U_h,h_tm1))

		
		h = (1-z)*hh + z*h_tm1

		tiled_mask_t = tf.tile(mask_t, tf.stack([1, tf.shape(h)[1]]))
		h = tf.where(tiled_mask_t, h, h_tm1)

		hidden_state.write(time, h)

		return (time+1,hidden_state,h)

	


	time = tf.constant(0, dtype='int32', name='time_q')


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
	output = tf.transpose(hidden_state,perm=axis)

	#linear classification
	W_c = init_weight_variable((output_dims,num_class),name="W_c_q")
	b_c = init_bias_variable((num_class,),name="b_r_q")
	# call softmax function
	# scores = tf.nn.softmax(tf.matmul(last_output,W_c)+b_c)

	scores = tf.matmul(last_output,W_c)+b_c
	return scores

if __name__=='__main__':
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

	print('test question emcording ...')

	size_voc = 10
	timesteps=10
	word_embedding_size = 10
	words = [1,2,3,4,5]
	num_class = 101
	question_embedding_size = 100

	input_question = tf.placeholder(tf.int32, shape=(None,timesteps), name='input_question')
	y = tf.placeholder(tf.float32,shape=(None, num_class))

	embeded_words, mask = getEmbedding(input_question, size_voc, word_embedding_size)
	embeded_question = getQuestionEncoder(embeded_words, question_embedding_size, mask, num_class)
	# sess.run(init)
	# train module
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = embeded_question))
	acc_value = tf.metrics.accuracy(y, embeded_question)
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)

	# runtime environment 
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	with sess.as_default():
		for i in range(1):
			data_x = np.random.randint(0,10,size=(2,timesteps),dtype='int32')
			data_y = np.zeros((2,num_class))
			data_y[:,1]=1
			_, l, output_embed,mask_p = sess.run([train,loss,embeded_words,mask],feed_dict={input_question:data_x,y:data_y})
			print(l)
			print(data_x)
			print(mask_p)


	


	

	# node1 = tf.constant(3.0, tf.float32)
	# node2 = tf.constant(4.0, tf.float32)

	# print(node1,node2)

	# sess = tf.Session()
	# print(sess.run([node1,node2]))

	# node3 = tf.add(node1,node2)
	# print("node3:", node3)
	# print("sess.run(node3):" , sess.run(node3))


	# a = tf.placeholder(tf.float32)
	# b = tf.placeholder(tf.float32)
	# adder_node = a + b

	# print(sess.run(adder_node,{a:3, b:4.5}))

	# print(sess.run(adder_node, {a:[1,3],b:[2,4]}))

	# W = tf.Variable([.3], tf.float32)
	# b = tf.Variable([-.3], tf.float32)

	# x  = tf.placeholder(tf.float32)
	# linear_model = W*x +b
	# # initialize all the variable in a TensorFlwo program
	# init = tf.global_variables_initializer()
	# sess.run(init)

	# print(sess.run(linear_model,{x:[1,2,3,4]}))

	# # the loss module
	# y = tf.placeholder(tf.float32)
	# squared_deltas = tf.square(linear_model - y)
	# loss = tf.reduce_sum(squared_deltas)
	# print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

	# # train module

	# optimizer = tf.train.GradientDescentOptimizer(0.01)
	# train = optimizer.minimize(loss)

	# sess.run(init)
	# for i in range(1000):
	# 	sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
	# print(sess.run([W,b]))


