import tensorflow as tf

import numpy as np

# def conv2d(x, W, strides=[1,1,1,1], padding='SAME'):
# 	return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def init_weight_variable(shape,name=None):
	initial = tf.truncated_normal(shape, stddev=0.1, name=name)
	return tf.Variable(initial)


def init_bias_variable(shape,name=None):
	initial = tf.constant(0.1,shape=shape, name=name)
	return tf.Variable(initial)


def matmul_wx(x, w, b, output_dims):
	return tf.matmul(x, w)+tf.reshape(b,(1,output_dims))

def matmul_uh(u,h_tm1):
	return tf.matmul(h_tm1,u)
'''
	x: batch_size, timesteps , dims
	num_class: number of class : ucf-101: 101
'''
def getModel(x, timesteps, input_dims, output_dims, num_class):

	

	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = init_weight_variable((input_dims,output_dims),name="W_r")
	W_z = init_weight_variable((input_dims,output_dims),name="W_z")
	W_h = init_weight_variable((input_dims,output_dims),name="W_h")

	U_r = init_weight_variable((output_dims,output_dims),name="U_r")
	U_z = init_weight_variable((output_dims,output_dims),name="U_z")
	U_h = init_weight_variable((output_dims,output_dims),name="U_h")

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

	initial_state = tf.zeros((100,output_dims),dtype=tf.float32)


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



if __name__=='__main__':
	timesteps=10
	input_dims=100
	output_dims=100
	num_class=10
	print('test..')
	input_x = tf.placeholder(tf.float32, shape=(None, timesteps, input_dims),name='input_x')
	y = tf.placeholder(tf.float32,shape=(None, num_class))

	scores = getModel(input_x, timesteps, input_dims, output_dims, num_class)

	
	# sess.run(init)

	


	# train module
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = scores))
	acc_value = tf.metrics.accuracy(y, scores)
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)

	# runtime environment 
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	with sess.as_default():
		for i in range(1000):
			data_x = np.random.random((100,timesteps,input_dims))
			data_y = np.zeros((100,num_class))
			data_y[:,1]=1
			_, l = sess.run([train,loss],feed_dict={input_x:data_x,y:data_y})
			print(l)

	print('done..')

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


