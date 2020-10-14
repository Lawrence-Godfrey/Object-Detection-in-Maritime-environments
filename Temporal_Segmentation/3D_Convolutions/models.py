from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, SpatialDropout3D
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import MaxPooling2D, MaxPool3D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
	"""Function to add 2 convolutional layers with the parameters passed to it"""
	# first layer
	x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(input_tensor)

	if batchnorm:
		x = BatchNormalization()(x)

	x = Activation('relu')(x)

	# second layer
	x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(input_tensor)

	if batchnorm:
		x = BatchNormalization()(x)

	x = Activation('relu')(x)
    
	return x	


def conv3d_block(input_tensor, n_filters, kernel_size = (3, 3, 3), batchnorm = True):
	"""Function to add 2 convolutional layers with the parameters passed to it"""
	# first layer
	x = Conv3D(filters = n_filters, kernel_size = (3, 3, 3), padding='same')(input_tensor)
	
	if batchnorm:
		x = BatchNormalization()(x)

	x = Activation('relu')(x)
	
	#Second Layer
	x = Conv3D(filters = n_filters, kernel_size = (3, 3, 3), padding='same')(x)
    
	if batchnorm:
		x = BatchNormalization()(x)

	x = Activation('relu')(x)

	return x	



def Unet2D(input_shape, n_filters = 16, dropout = 0.1, batchnorm = True):
	input_layer = Input(input_shape, name='input')

	# Contracting Path
	c1 = conv2d_block(input_layer, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
	p1 = MaxPooling2D((2, 2))(c1)
	p1 = Dropout(dropout)(p1)

	c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
	p2 = MaxPooling2D((2, 2))(c2)
	p2 = Dropout(dropout)(p2)

	c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
	p3 = MaxPooling2D((2, 2))(c3)
	p3 = Dropout(dropout)(p3)

	c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
	p4 = MaxPooling2D((2, 2))(c4)
	p4 = Dropout(dropout)(p4)

	c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

	# Expansive Path
	u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
	u6 = concatenate([u6, c4])
	u6 = Dropout(dropout)(u6)
	c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

	u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
	u7 = concatenate([u7, c3])
	u7 = Dropout(dropout)(u7)
	c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

	u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
	u8 = concatenate([u8, c2])
	u8 = Dropout(dropout)(u8)
	c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

	u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
	u9 = concatenate([u9, c1])
	u9 = Dropout(dropout)(u9)
	c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

	outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
	model = Model(inputs=[input_layer], outputs=[outputs])
	return model




def Unet3D(input_shape, n_filters = 16, dropout = 0.1, batchnorm=True):
	input_layer = Input(input_shape, name='input')

	# Contracting Path
	c0 = Conv3D(filters = n_filters * 1, kernel_size = (3, 3, 3), padding='same')(input_layer)
	if batchnorm:
		c0 = BatchNormalization()(c0)
	c0 = Activation('relu')(c0)
	p0 = MaxPool3D((2, 2, 1))(c0)
	p0 = SpatialDropout3D(dropout)(p0)

	c1 = Conv3D(filters = n_filters * 1, kernel_size = (3, 3, 3), padding='same')(p0)
	if batchnorm:
		c1 = BatchNormalization()(c1)
	c1 = Activation('relu')(c1)
	p1 = MaxPool3D((2, 2, 1))(c1)
	p1 = SpatialDropout3D(dropout)(p1)

	c2 = conv3d_block(p1, n_filters * 2, batchnorm=batchnorm)
	p2 = MaxPool3D((2, 2, 1))(c2)
	p2 = SpatialDropout3D(dropout)(p2)

	c3 = conv3d_block(p2, n_filters * 4, batchnorm=batchnorm)
	p3 = MaxPool3D((2, 2, 2))(c3)
	p3 = SpatialDropout3D(dropout)(p3)

	c4 = conv3d_block(p3, n_filters * 8, batchnorm=batchnorm)
	p4 = MaxPool3D((2, 2, 2))(c4)
	p4 = SpatialDropout3D(dropout)(p4)

	c5 = conv3d_block(p4, n_filters = n_filters * 16, batchnorm=batchnorm)

	# Expansive Path
	u6 = Conv3DTranspose(n_filters * 8, (3, 3, 3), strides=(2, 2, 2), padding = 'SAME')(c5)
	u6 = concatenate([u6, c4])
	u6 = SpatialDropout3D(dropout)(u6)
	c6 = conv3d_block(u6, n_filters * 8, batchnorm=batchnorm)

	u7 = Conv3DTranspose(n_filters * 4, (3, 3, 3), strides=(2, 2, 2), padding = 'SAME')(c6)
	u7 = concatenate([u7, c3])
	u7 = SpatialDropout3D(dropout)(u7)
	c7 = conv3d_block(u7, n_filters * 4, batchnorm=batchnorm)

	u8 = Conv3DTranspose(n_filters * 2, (3, 3, 3), strides=(2, 2, 1), padding = 'SAME')(c7)
	u8 = concatenate([u8, c2])
	u8 = SpatialDropout3D(dropout)(u8)
	c8 = conv3d_block(u8, n_filters * 2, batchnorm=batchnorm)

	u9 = Conv3DTranspose(n_filters * 1, (3, 3, 3), strides=(2, 2, 1), padding = 'SAME')(c8)
	u9 = concatenate([u9, c1])
	u9 = SpatialDropout3D(dropout)(u9)
	c9 = conv3d_block(u9, n_filters * 1, batchnorm=batchnorm)

	u10 = Conv3DTranspose(n_filters * 1, (3, 3, 3), strides=(2, 2, 1), padding = 'SAME')(c9)
	u10 = concatenate([u10, c0])
	u10 = SpatialDropout3D(dropout)(u10)
	c10 = conv3d_block(u10, n_filters * 1, batchnorm=batchnorm)

	outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c10)
	print(outputs[..., 0].shape)
	model = Model(inputs=[input_layer], outputs=[outputs[..., 0]])
	return model

