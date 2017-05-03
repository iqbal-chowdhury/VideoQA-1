import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import numpy as np
from sklearn.decomposition import PCA
import ModelUtil
import InitUtil


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(input)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    if isinstance(x, tf.SparseTensor):
        return x._dims

    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None
def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x, y: Keras tensors or variables with `ndim >= 2`
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
        of `x.dot(y.T)`, although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
        If `axes` is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in `x`'s shape and `y`'s shape:

        * `x.shape[0]` : 100 : append to output shape
        * `x.shape[1]` : 20 : do not append to output shape,
            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
        * `y.shape[0]` : 100 : do not append to output shape,
            always ignore first dimension of `y`
        * `y.shape[1]` : 30 : append to output shape
        * `y.shape[2]` : 20 : do not append to output shape,
            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
        `output_shape` = `(100, 30)`

    ```python
        >>> x_batch = K.ones(shape=(32, 20, 1))
        >>> y_batch = K.ones(shape=(32, 30, 20))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    if ndim(x) == 2 and ndim(y) == 2:
        if tf_major_version >= 1:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.multiply(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
        else:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.mul(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.mul(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        # TODO: remove later.
        if hasattr(tf, 'batch_matmul'):
            try:
                out = tf.batch_matmul(x, y, adj_a=adj_x, adj_b=adj_y)
            except TypeError:
                out = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
        else:
            out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out

def getVideoDualSemanticEmbedding(x,w2v,embedded_stories_words,T_B,pca_mat=None):
	'''
		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
		w2v: word 2 vec (|v|,dim)
	'''
	input_shape = x.get_shape().as_list()
	w2v_shape = w2v.get_shape().as_list()
	assert(len(input_shape)==5)
	axis = [0,1,3,4,2]
	x = tf.transpose(x,perm=axis)
	x = tf.reshape(x,(-1,input_shape[2]))
	# x = tf.nn.l2_normalize(x,-1)

	if pca_mat is not None:
		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	x = tf.matmul(x,linear_proj) 
	x = tf.nn.l2_normalize(x,-1)

	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)

	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
	axis = [0,1,4,2,3]
	x = tf.transpose(x,perm=axis)
	
	# can be extended to different architecture
	x = tf.reduce_sum(x,reduction_indices=[3,4])
	x = tf.nn.l2_normalize(x,-1)


	stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
	print('stories_cov.get_shape():',stories_cov.get_shape().as_list())
	x = batch_dot(x,stories_cov)
	print('x.get_shape():',x.get_shape().as_list())
	x = tf.reshape(x,(-1,w2v_shape[-1]))
	

	x = tf.matmul(x,T_B)
	x = tf.reshape(x,(-1,input_shape[1],w2v_shape[-1]))
	x = tf.reduce_sum(x,reduction_indices=[1])
	x = tf.nn.l2_normalize(x,-1)
	return x


def getVideoDualSemanticEmbeddingWithQuestionAttention(x,w2v,embedded_stories_words,embedded_question,T_B,pca_mat=None):
	'''
		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
		w2v: word 2 vec (|v|,dim)
	'''
	input_shape = x.get_shape().as_list()
	w2v_shape = w2v.get_shape().as_list()
	assert(len(input_shape)==5)
	axis = [0,1,3,4,2]
	x = tf.transpose(x,perm=axis)
	x = tf.reshape(x,(-1,input_shape[2]))
	# x = tf.nn.l2_normalize(x,-1)

	if pca_mat is not None:
		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	x = tf.matmul(x,linear_proj) 
	x = tf.nn.l2_normalize(x,-1)


	#-----------------------
	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	
	#-----------------------

	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
	axis = [0,1,4,2,3]
	x = tf.transpose(x,perm=axis)
	
	# can be extended to different architecture
	x = tf.reduce_sum(x,reduction_indices=[3,4])
	x = tf.nn.l2_normalize(x,-1)

	#-----------------------

	stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
	x = batch_dot(x,stories_cov)
	#-----------------------
	x = tf.nn.l2_normalize(x,-1)

	embedded_question = tf.tile(tf.expand_dims(embedded_question,dim=1),[1,input_shape[1],1])

	
	

	
	frame_weight = tf.reduce_sum(x*embedded_question,reduction_indices=-1,keep_dims=True)
	frame_weight = tf.nn.softmax(frame_weight,dim=1)

	frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

	x = tf.reduce_sum(x*frame_weight,reduction_indices=1)
	# x = tf.nn.l2_normalize(x,-1)
	x = tf.matmul(x,T_B)

	x = tf.nn.l2_normalize(x,-1)
	return x


def getAverageRepresentation(sentence, T_B, d_lproj):
	sentence = tf.reduce_sum(sentence,reduction_indices=-2)

	sentence_shape = sentence.get_shape().as_list()
	if len(sentence_shape)==2:
		sentence = tf.matmul(sentence,T_B)
	elif len(sentence_shape)==3:
		sentence = tf.reshape(sentence,(-1,sentence_shape[-1]))
		sentence = tf.matmul(sentence,T_B)
		sentence = tf.reshape(sentence,(-1,sentence_shape[1],d_lproj))
	else:
		raise ValueError('Invalid sentence_shape:'+sentence_shape)

	sentence = tf.nn.l2_normalize(sentence,-1)
	return sentence




def getVideoSemanticEmbedding(x,w2v,T_B,pca_mat=None):
	'''
		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
		w2v: word 2 vec (|v|,dim)
	'''
	input_shape = x.get_shape().as_list()
	w2v_shape = w2v.get_shape().as_list()
	assert(len(input_shape)==5)
	axis = [0,1,3,4,2]
	x = tf.transpose(x,perm=axis)
	x = tf.reshape(x,(-1,input_shape[2]))
	# x = tf.nn.l2_normalize(x,-1)

	if pca_mat is not None:
		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	x = tf.matmul(x,linear_proj) 
	x = tf.nn.l2_normalize(x,-1)

	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)

	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
	axis = [0,1,4,2,3]
	x = tf.transpose(x,perm=axis)
	
	# can be extended to different architecture
	x = tf.reduce_sum(x,reduction_indices=[1,3,4])
	x = tf.nn.l2_normalize(x,-1)

	x = tf.matmul(x,T_B)

	return x


# def getVideoHierarchicalSemantic_ClipLevel(x,w2v,embedded_stories_words,embedded_question,T_B,pca_mat=None):
# 	'''
# 		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
# 		w2v: word 2 vec (|v|,dim)
# 	'''
# 	input_shape = x.get_shape().as_list()
# 	w2v_shape = w2v.get_shape().as_list()
# 	assert(len(input_shape)==5)
# 	axis = [0,1,3,4,2]
# 	x = tf.transpose(x,perm=axis)
# 	x = tf.reshape(x,(-1,input_shape[2]))
# 	# x = tf.nn.l2_normalize(x,-1)

# 	if pca_mat is not None:
# 		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
# 	else:
# 		linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

# 	x = tf.matmul(x,linear_proj) 
# 	x = tf.nn.l2_normalize(x,-1)


# 	#-----------------------
# 	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
# 	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	
# 	#-----------------------

# 	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
# 	axis = [0,1,4,2,3]
# 	x = tf.transpose(x,perm=axis)
	
# 	# can be extended to different architecture
# 	x = tf.reduce_sum(x,reduction_indices=[3,4])
# 	x = tf.nn.l2_normalize(x,-1)

# 	#-----------------------

# 	stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
# 	x = batch_dot(x,stories_cov)
# 	#-----------------------
# 	x = tf.nn.l2_normalize(x,-1)

# 	embedded_question = tf.tile(tf.expand_dims(embedded_question,dim=1),[1,input_shape[1],1])

	
	

	
# 	frame_weight = tf.reduce_sum(x*embedded_question,reduction_indices=-1,keep_dims=True)
# 	frame_weight = tf.nn.softmax(frame_weight,dim=1)

# 	frame_weight =tf.tile(frame_weight,[1,1,w2v_shape[-1]])

# 	x = x*frame_weight
# 	# x = tf.reshape(x,(-1,w2v_shape[-1]))
	
# 	# x = tf.matmul(x,T_B)
# 	# x = tf.reshape(x,(-1,input_shape[1],w2v_shape[-1]))
# 	# x = tf.nn.l2_normalize(x,-1)
# 	return x

def hard_sigmoid(x):
	x = (0.2 * x) + 0.5
	x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),tf.cast(1., dtype=tf.float32))
	return x

def get_init_state(x, output_dims):
	initial_state = tf.zeros_like(x)
	initial_state = tf.reduce_sum(initial_state,axis=[1,2])
	initial_state = tf.expand_dims(initial_state,dim=-1)
	initial_state = tf.tile(initial_state,[1,output_dims])
	return initial_state
# def getAnswerGRUEncoder(embeded_words, output_dims, mask, return_sequences=False):
	
# 	input_shape = embeded_words.get_shape().as_list()
# 	assert len(input_shape)==4 

# 	numberOfChoices = input_shape[1]
# 	timesteps = input_shape[2]
# 	input_dims = input_shape[3]

# 	# get initial state
# 	embeded_words = tf.reshape(embeded_words,(-1,timesteps,input_dims))
# 	initial_state = get_init_state(embeded_words, output_dims)

# 	axis = [1,0,2]  
# 	embeded_words = tf.transpose(embeded_words, perm=axis) # permutate the 'embeded_words' --> timesteps x batch_size x numberOfChoices x dim
# 	# embeded_words = tf.reshape(embeded_words,(timesteps,-1,input_dims)) # reshape the 'embeded_words' --> timesteps x (batch x numberOfChoices) x dim
	
# 	# initialize the parameters
# 	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
# 	W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_r")
# 	W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_z")
# 	W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_a_h")

# 	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_r")
# 	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_z")
# 	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_a_h")

# 	b_r = InitUtil.init_bias_variable((output_dims,),name="b_a_r")
# 	b_z = InitUtil.init_bias_variable((output_dims,),name="b_a_z")
# 	b_h = InitUtil.init_bias_variable((output_dims,),name="b_a_h")



# 	input_embeded_words = tf.TensorArray(
#             dtype=embeded_words.dtype,
#             size=timesteps,
#             tensor_array_name='input_embeded_words_a')


# 	if hasattr(input_embeded_words, 'unstack'):
# 		input_embeded_words = input_embeded_words.unstack(embeded_words)
# 	else:
# 		input_embeded_words = input_embeded_words.unpack(embeded_words)	


# 	# preprocess mask
# 	if len(mask.get_shape()) == len(input_shape)-1:
# 		mask = tf.expand_dims(mask,dim=-1)
	
# 	axis = [2,0,1,3]  
# 	mask = tf.transpose(mask,perm=axis)
# 	mask = tf.reshape(mask, (timesteps,-1,1))

# 	input_mask = tf.TensorArray(
# 		dtype=mask.dtype,
# 		size=timesteps,
# 		tensor_array_name='input_mask_q'
# 		)

# 	if hasattr(input_mask, 'unstack'):
# 		input_mask = input_mask.unstack(mask)
# 	else:
# 		input_mask = input_mask.unpack(mask)


# 	hidden_state_q = tf.TensorArray(
#             dtype=tf.float32,
#             size=timesteps,
#             tensor_array_name='hidden_state_a')

# 	# if hasattr(hidden_state, 'unstack'):
# 	# 	hidden_state = hidden_state.unstack(hidden_state)
# 	# else:
# 	# 	hidden_state = hidden_state.unpack(hidden_state)


# 	def step(time, hidden_state_q, h_tm1):
# 		x_t = input_embeded_words.read(time) # batch_size * dim
# 		mask_t = input_mask.read(time)

# 		preprocess_x_r = tf.nn.xw_plus_b(x_t, W_r, b_r)
# 		preprocess_x_z = tf.nn.xw_plus_b(x_t, W_z, b_z)
# 		preprocess_x_h = tf.nn.xw_plus_b(x_t, W_h, b_h)

# 		r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1,U_r) )
# 		z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1,U_z) )
# 		hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1,U_h) )
		
# 		h = (1-z)*hh + z*h_tm1
# 		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

# 		h = tf.where(tiled_mask_t, h, h_tm1)
		
# 		hidden_state_q = hidden_state_q.write(time, h)

# 		return (time+1,hidden_state_q,h)

	


# 	time = tf.constant(0, dtype='int32', name='time')


# 	ret_out = tf.while_loop(
#             cond=lambda time, *_: time < timesteps,
#             body=step,
#             loop_vars=(time, hidden_state_q, initial_state),
#             parallel_iterations=32,
#             swap_memory=True)


# 	hidden_state_q = ret_out[1]
# 	last_output = ret_out[-1] 


	
# 	if hasattr(hidden_state_q, 'stack'):
# 		outputs = hidden_state_q.stack()
# 		print('stack')
# 	else:
# 		outputs = hidden_state_q.pack()

# 	outputs = tf.reshape(outputs,(timesteps,-1,numberOfChoices,output_dims))
# 	axis = [1,2,0]+list(range(3,4))
# 	outputs = tf.transpose(outputs,perm=axis)

# 	last_output = tf.reshape(last_output,(-1,numberOfChoices,output_dims))
# 	print('outputs:....',outputs.get_shape().as_list())
# 	if return_sequences:
# 		return outputs
# 	else:
# 		return last_output


# def getQuestionGRUEncoderWithAttention(embeded_words, embedded_video, output_dims, mask, return_sequences=False):

# 	'''
# 		function: getQuestionEncoder
# 		parameters:
# 			embeded_words: sample*timestep*dim
# 			output_dims: the GRU hidden dim
# 			mask: bool type , samples * timestep
# 		return:
# 			the last GRU state, 
# 			or
# 			the sequences of the hidden states
# 	'''
# 	input_shape = embeded_words.get_shape().as_list()
# 	assert len(input_shape)==3 

# 	timesteps = input_shape[1]
# 	input_dims = input_shape[2]
# 	# get initial state
# 	initial_state = get_init_state(embeded_words, output_dims)


# 	# initialize the parameters
# 	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
# 	W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_r")
# 	W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_z")
# 	W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_q_h")

# 	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_r")
# 	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_z")
# 	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_q_h")

# 	b_r = InitUtil.init_bias_variable((output_dims,),name="b_q_r")
# 	b_z = InitUtil.init_bias_variable((output_dims,),name="b_q_z")
# 	b_h = InitUtil.init_bias_variable((output_dims,),name="b_q_h")


# 	# attention
# 	attention_dim = 100
# 	W_a = InitUtil.init_weight_variable((input_dims,attention_dim),init_method='glorot_uniform',name="W_a")
# 	U_a = InitUtil.init_weight_variable((output_dims,attention_dim),init_method='orthogonal',name="U_a")
# 	b_a = InitUtil.init_bias_variable((attention_dim,),name="b_a")

# 	W = InitUtil.init_weight_variable((attention_dim,1),init_method='glorot_uniform',name="W")

# 	A_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_z")

# 	A_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_r")

# 	A_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_h")



# 	# batch_size x timesteps x dim -> timesteps x batch_size x dim
# 	axis = [1,0]+list(range(2,3))  # axis = [1,0,2]
# 	embeded_words = tf.transpose(embeded_words, perm=axis) # permutate the input_x --> timestemp, batch_size, input_dims



# 	input_embeded_words = tf.TensorArray(
#             dtype=embeded_words.dtype,
#             size=timesteps,
#             tensor_array_name='input_embeded_words_q')


# 	if hasattr(input_embeded_words, 'unstack'):
# 		input_embeded_words = input_embeded_words.unstack(embeded_words)
# 	else:
# 		input_embeded_words = input_embeded_words.unpack(embeded_words)	


# 	# preprocess mask
# 	if len(mask.get_shape()) == len(input_shape)-1:
# 		mask = tf.expand_dims(mask,dim=-1)
	
# 	mask = tf.transpose(mask,perm=axis)

# 	input_mask = tf.TensorArray(
# 		dtype=mask.dtype,
# 		size=timesteps,
# 		tensor_array_name='input_mask_q'
# 		)

# 	if hasattr(input_mask, 'unstack'):
# 		input_mask = input_mask.unstack(mask)
# 	else:
# 		input_mask = input_mask.unpack(mask)


# 	hidden_state_q = tf.TensorArray(
#             dtype=tf.float32,
#             size=timesteps,
#             tensor_array_name='hidden_state_q')

# 	timesteps_v = embedded_video.get_shape().as_list()[1]
# 	print('timesteps_v:',timesteps_v)
# 	def step(time, hidden_state_q, embedded_video, h_tm1):
# 		x_t = input_embeded_words.read(time) # batch_size * dim
# 		mask_t = input_mask.read(time)


# 		ori_feature = tf.reshape(embedded_video,(-1,input_dims))
# 		attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, W_a, b_a),(-1,timesteps_v,attention_dim))
# 		attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, U_a),dim=1),[1,timesteps_v,1])

# 		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
# 		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)
# 		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_v,1)),dim=1)

# 		attend_fea = embedded_video * tf.tile(attend_e,[1,1,input_dims])
# 		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



# 		preprocess_x_r = tf.nn.xw_plus_b(x_t, W_r, b_r)
# 		preprocess_x_z = tf.nn.xw_plus_b(x_t, W_z, b_z)
# 		preprocess_x_h = tf.nn.xw_plus_b(x_t, W_h, b_h)

# 		r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1,U_r) + tf.matmul(attend_fea,A_r))
# 		z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1,U_z) + tf.matmul(attend_fea,A_z))
# 		hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1,U_h) + tf.matmul(attend_fea,A_h))

		
# 		h = (1-z)*hh + z*h_tm1
# 		tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

# 		h = tf.where(tiled_mask_t, h, h_tm1)
		
# 		hidden_state_q = hidden_state_q.write(time, h)

# 		return (time+1,hidden_state_q, embedded_video, h)

	


# 	time = tf.constant(0, dtype='int32', name='time')


# 	ret_out = tf.while_loop(
#             cond=lambda time, *_: time < timesteps,
#             body=step,
#             loop_vars=(time, hidden_state_q, embedded_video, initial_state),
#             parallel_iterations=32,
#             swap_memory=True)


# 	hidden_state_q = ret_out[1]
# 	last_output = ret_out[-1] 
	
# 	if hasattr(hidden_state_q, 'stack'):
# 		outputs = hidden_state_q.stack()
# 		print('stack')
# 	else:
# 		outputs = hidden_state_q.pack()

# 	axis = [1,0] + list(range(2,3))
# 	outputs = tf.transpose(outputs,perm=axis)

# 	if return_sequences:
# 		return outputs
# 	else:
# 		return last_output

def getMemoryNetworks(embeded_stories, embeded_question, d_lproj, return_sequences=False):

	'''
		embeded_stories: (batch_size, num_of_sentence, num_of_words, embeded_words_dims)
		embeded_question:(batch_size, embeded_words_dims)
		output_dims: the dimension of stories 
	'''
	stories_shape = embeded_stories.get_shape().as_list()
	embeded_question_shape = embeded_question.get_shape().as_list()
	num_of_sentence = stories_shape[-3]
	input_dims = stories_shape[-1]
	output_dims = embeded_question_shape[-1]


	embeded_stories = tf.reduce_sum(embeded_stories,reduction_indices=-2)
	embeded_stories = tf.nn.l2_normalize(embeded_stories,-2)

	
	embeded_question = tf.tile(tf.expand_dims(embeded_question,dim=1),[1,num_of_sentence,1])

	sen_weight = tf.reduce_sum(embeded_question*embeded_stories,reduction_indices=-1,keep_dims=True)

	sen_weight = tf.nn.softmax(sen_weight,dim=1)
	sen_weight = tf.tile(sen_weight,[1,1,output_dims])
	if return_sequences:
		embeded_stories = embeded_stories*sen_weight
	else:
		embeded_stories = tf.reduce_sum(embeded_stories*sen_weight,reduction_indices=1) # (batch_size, output_dims)

	return embeded_stories


def getVideoHierarchicalSemanticWithAttendQuestion(x,w2v,embedded_stories_words,embedded_question,T_B,pca_mat=None, return_sequences=False):
	'''
		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
		w2v: word 2 vec (|v|,dim)
	'''
	input_shape = x.get_shape().as_list()
	w2v_shape = w2v.get_shape().as_list()
	assert(len(input_shape)==5)
	axis = [0,1,3,4,2]
	x = tf.transpose(x,perm=axis)
	x = tf.reshape(x,(-1,input_shape[2]))
	# x = tf.nn.l2_normalize(x,-1)

	if pca_mat is not None:
		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	x = tf.matmul(x,linear_proj) 
	x = tf.nn.l2_normalize(x,-1)


	#-----------------------
	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	
	#-----------------------

	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
	axis = [0,1,4,2,3]
	x = tf.transpose(x,perm=axis)
	
	# can be extended to different architecture
	x = tf.reduce_sum(x,reduction_indices=[3,4])
	x = tf.nn.l2_normalize(x,-1)

	#-----------------------

	stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
	x = batch_dot(x,stories_cov)
	#-----------------------
	x = tf.nn.l2_normalize(x,-1)


	input_shape = x.get_shape().as_list()
	assert len(input_shape)==3 

	timesteps = input_shape[1]
	input_dims = input_shape[2]
	output_dims = input_dims
# 	# get initial state
	initial_state = get_init_state(x, output_dims)


	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_r")
	W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_z")
	W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_h")

	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_r")
	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_z")
	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_h")

	b_r = InitUtil.init_bias_variable((output_dims,),name="b_v_r")
	b_z = InitUtil.init_bias_variable((output_dims,),name="b_v_z")
	b_h = InitUtil.init_bias_variable((output_dims,),name="b_v_h")




# 	# batch_size x timesteps x dim -> timesteps x batch_size x dim
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

	hidden_state_x = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_x')

	def step(time, hidden_state_x, h_tm1):
		x_t = input_x.read(time) # batch_size * dim


		preprocess_x_r = tf.nn.xw_plus_b(x_t, W_r, b_r)
		preprocess_x_z = tf.nn.xw_plus_b(x_t, W_z, b_z)
		preprocess_x_h = tf.nn.xw_plus_b(x_t, W_h, b_h)

		r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1,U_r))
		z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1,U_z))
		hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1,U_h))

		h = (1-z)*hh + z*h_tm1
		
		hidden_state_x = hidden_state_x.write(time, h)

		return (time+1,hidden_state_x, h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state_x, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	hidden_state_x = ret_out[1]
	last_output = ret_out[-1] 
	
	if hasattr(hidden_state_x, 'stack'):
		outputs = hidden_state_x.stack()
		print('stack')
	else:
		outputs = hidden_state_x.pack()

	axis = [1,0] + list(range(2,3))
	outputs = tf.transpose(outputs,perm=axis)
	last_output = tf.matmul(last_output,T_B)
	last_output = tf.nn.l2_normalize(last_output,-1)
	if return_sequences:
		return outputs
	else:
		return last_output





def getVideoHierarchicalSemanticWithAttendQuestionExe(x,w2v,embedded_stories_words,embedded_question,T_B, mask_q, pca_mat=None, return_sequences=False):
	'''
		x: input video cnn feature with size of (batch_size, timesteps, channels, height, width)
		w2v: word 2 vec (|v|,dim)
	'''
	input_shape = x.get_shape().as_list()
	w2v_shape = w2v.get_shape().as_list()
	assert(len(input_shape)==5)
	axis = [0,1,3,4,2]
	x = tf.transpose(x,perm=axis)
	x = tf.reshape(x,(-1,input_shape[2]))
	# x = tf.nn.l2_normalize(x,-1)

	if pca_mat is not None:
		linear_proj = tf.Variable(0.1*pca_mat,dtype='float32',name='visual_linear_proj')
	else:
		linear_proj = InitUtil.init_weight_variable((input_shape[2],w2v_shape[-1]), init_method='uniform', name='visual_linear_proj')

	x = tf.matmul(x,linear_proj) 
	x = tf.nn.l2_normalize(x,-1)


	#-----------------------
	w2v_cov = tf.matmul(tf.transpose(w2v,perm=[1,0]),w2v)
	x = tf.matmul(x,w2v_cov) # (batch_size*timesteps*height*width, |V|)

	
	#-----------------------

	x = tf.reshape(x,(-1,input_shape[1],input_shape[3],input_shape[4],w2v_shape[-1]))
	axis = [0,1,4,2,3]
	x = tf.transpose(x,perm=axis)
	
	# can be extended to different architecture
	x = tf.reduce_sum(x,reduction_indices=[3,4])
	x = tf.nn.l2_normalize(x,-1)

	#-----------------------

	# embedded_stories_words = tf.nn.l2_normalize(embedded_stories_words,-1)### test

	stories_cov = batch_dot(tf.transpose(embedded_stories_words,perm=[0,2,1]),embedded_stories_words)
	x = batch_dot(x,stories_cov)
	#-----------------------
	# x = tf.nn.l2_normalize(x,-1)


	input_shape = x.get_shape().as_list()
	assert len(input_shape)==3 

	timesteps = input_shape[1]
	input_dims = input_shape[2]
	output_dims = input_dims
# 	# get initial state
	initial_state = get_init_state(x, output_dims)


	# initialize the parameters
	# W_r,U_r,b_r; W_z, U_z, b_z; W_h, U_h, b_h
	# W_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_r")
	# W_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_z")
	# W_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='glorot_uniform',name="W_v_h")

	U_r = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_r")
	U_z = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_z")
	U_h = InitUtil.init_weight_variable((output_dims,output_dims),init_method='orthogonal',name="U_v_h")

	# b_r = InitUtil.init_bias_variable((output_dims,),name="b_v_r")
	# b_z = InitUtil.init_bias_variable((output_dims,),name="b_v_z")
	# b_h = InitUtil.init_bias_variable((output_dims,),name="b_v_h")

	# attention
	attention_dim = 30
	W_a = InitUtil.init_weight_variable((input_dims,attention_dim),init_method='glorot_uniform',name="W_a")
	U_a = InitUtil.init_weight_variable((output_dims,attention_dim),init_method='orthogonal',name="U_a")
	b_a = InitUtil.init_bias_variable((attention_dim,),name="b_a")

	W = InitUtil.init_weight_variable((attention_dim,1),init_method='glorot_uniform',name="W")

	A_z = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_z")

	A_r = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_r")

	A_h = InitUtil.init_weight_variable((input_dims,output_dims),init_method='orthogonal',name="A_h")





# 	# batch_size x timesteps x dim -> timesteps x batch_size x dim
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

	hidden_state_x = tf.TensorArray(
            dtype=tf.float32,
            size=timesteps,
            tensor_array_name='hidden_state_x')

	timesteps_q = embedded_question.get_shape().as_list()[1]


	def step(time, hidden_state_x, h_tm1):
		x_t = input_x.read(time) # batch_size * dim

		ori_feature = tf.reshape(embedded_question,(-1,input_dims))
		attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, W_a, b_a),(-1,timesteps_q,attention_dim))
		attend_uh_tm1 = tf.tile(tf.expand_dims(tf.matmul(h_tm1, U_a),dim=1),[1,timesteps_q,1])

		attend_e = tf.nn.tanh(attend_wx+attend_uh_tm1)
		attend_e = tf.matmul(tf.reshape(attend_e,(-1,attention_dim)),W)# batch_size * timestep
		# attend_e = tf.reshape(attend_e,(-1,attention_dim))
		attend_e = tf.where(tf.reshape(mask_q,(-1,)), attend_e, tf.zeros_like(attend_e))
		attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,timesteps_q,1)),dim=1)

		attend_fea = embedded_question * tf.tile(attend_e,[1,1,input_dims])
		attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)



		# preprocess_x_r = tf.nn.xw_plus_b(x_t, W_r, b_r)
		# preprocess_x_z = tf.nn.xw_plus_b(x_t, W_z, b_z)
		# preprocess_x_h = tf.nn.xw_plus_b(x_t, W_h, b_h)

		r = hard_sigmoid(x_t+ tf.matmul(h_tm1,U_r) + tf.matmul(attend_fea,A_r))
		z = hard_sigmoid(x_t+ tf.matmul(h_tm1,U_z) + tf.matmul(attend_fea,A_z))
		hh = tf.nn.tanh(x_t+ tf.matmul(r*h_tm1,U_h) + tf.matmul(attend_fea,A_h))

		h = (1-z)*hh + z*h_tm1
		
		hidden_state_x = hidden_state_x.write(time, h)

		return (time+1,hidden_state_x, h)

	


	time = tf.constant(0, dtype='int32', name='time')


	ret_out = tf.while_loop(
            cond=lambda time, *_: time < timesteps,
            body=step,
            loop_vars=(time, hidden_state_x, initial_state),
            parallel_iterations=32,
            swap_memory=True)


	hidden_state_x = ret_out[1]
	last_output = ret_out[-1] 
	
	if hasattr(hidden_state_x, 'stack'):
		outputs = hidden_state_x.stack()
		print('stack')
	else:
		outputs = hidden_state_x.pack()

	axis = [1,0] + list(range(2,3))
	outputs = tf.transpose(outputs,perm=axis)

	if return_sequences:
		return outputs
	else:
		return last_output



