import numpy as np
import tensorflow as tf
import numpy as np

from models.attenvis import AttentionVis  
av = AttentionVis()

@av.att_mat_func
def Attention_Matrix(K, Q, use_mask=False):
	"""
	This functions runs a single attention head.

	:param K: is [batch_size x window_size_keys x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention matrix
	"""
	
	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	embedding_size = Q.get_shape()[2]
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])
	
	scores = tf.matmul(Q, tf.transpose(K, perm=[0,2,1])) / float(np.sqrt(embedding_size))
	if use_mask: scores = scores + atten_mask
	return tf.nn.softmax(scores)


class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):		
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask

		self.K = self.add_weight(shape=(input_size, output_size), initializer='random_normal', trainable=True)
		self.V = self.add_weight(shape=(input_size, output_size), initializer='random_normal', trainable=True)
		self.Q = self.add_weight(shape=(input_size, output_size), initializer='random_normal', trainable=True)
		
	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

		"""
		This functions runs a single attention head.

		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""
		K = tf.tensordot(inputs_for_keys, self.K, [[2],[0]])	# shape is [batch_size x window_size x output_size]
		V = tf.tensordot(inputs_for_values, self.V, [[2],[0]])	# shape is [batch_size x window_size x output_size]
		Q = tf.tensordot(inputs_for_queries, self.Q, [[2],[0]])	# shape is [batch_size x window_size x output_size]

		atten = Attention_Matrix(K, Q, self.use_mask)   # shape is [batch_size x window_size x window_size]
		
		values = tf.matmul(atten, V)	# shape is [batch_size x window_size x output_size]

		return values


class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		This functions creates a feed forward network as described in 3.3
		https://arxiv.org/pdf/1706.03762.pdf

		Requirements:
		- Two linear layers with relu between them

		:param inputs: input tensor [batch_size x window_size x embedding_size]
		:return: tensor [batch_size x window_size x embedding_size]
		"""
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder):
		super(Transformer_Block, self).__init__()

		self.ff_layer = Feed_Forwards(emb_sz)
		self.self_atten = Atten_Head(emb_sz,emb_sz,use_mask=is_decoder)
		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_atten = Atten_Head(emb_sz,emb_sz,use_mask=False)

		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None):
		"""
		This functions calls a transformer block.

		There are two possibilities for when this function is called.
			- if self.is_decoder == False, then:
				1) compute unmasked attention on the inputs
				2) residual connection and layer normalization
				3) feed forward layer
				4) residual connection and layer normalization

			- if self.is_decoder == True, then:
				1) compute MASKED attention on the inputs
				2) residual connection and layer normalization
				3) computed UNMASKED attention using context
				4) residual connection and layer normalization
				5) feed forward layer
				6) residual layer and layer normalization

		:param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""

		with av.trans_block(self.is_decoder):
			atten_out = self.self_atten(inputs,inputs,inputs)
		atten_out+=inputs
		atten_normalized = self.layer_norm(atten_out)

		if self.is_decoder:
			assert context is not None,"Decoder blocks require context"
			context_atten_out = self.self_context_atten(context,context,atten_normalized)
			context_atten_out+=atten_normalized
			atten_normalized = self.layer_norm(context_atten_out)

		ff_out=self.ff_layer(atten_normalized)
		ff_out+=atten_normalized
		ff_norm = self.layer_norm(ff_out)

		return tf.nn.relu(ff_norm)
