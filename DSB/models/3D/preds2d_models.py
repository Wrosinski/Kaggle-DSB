Stable!

def cnn_3d(width):
    optimizer = SGD(lr = 5e-5, momentum = 0.9, decay = 1e-3, nesterov = True)
    optimizer = Adam(lr = 5e-5)
    
    inputs = Input(shape=(1, 136, 168, 168))
    conv1 = Convolution3D(width, 3, 3, 3, activation = 'relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv1)
    
    conv2 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv2)

    conv3 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv3)
    
    conv4 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution3D(width*16, 3, 3, 3, activation = 'relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv4)
    
    conv5 = Convolution3D(width*32, 3, 3, 3, activation = 'relu', border_mode='same')(pool4)
    conv5 = BatchNormalization(axis = 1)(conv5)
    pool5 = MaxPooling3D(pool_size=(8, 8, 8), border_mode='same')(conv5)
    
    output = GlobalAveragePooling3D()(pool5)
    output = Dense(2, activation='softmax', name = 'predictions')(output)
    model3d = Model(inputs, output)
    model3d.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model3d
 
