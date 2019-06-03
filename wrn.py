import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation,AlphaDropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten,PReLU,GlobalAveragePooling2D,GlobalAveragePooling3D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from random_eraser import get_random_eraser  # added
from keras.utils import to_categorical
from keras import optimizers
from keras import losses
import numpy as np
from keras.engine.topology import Layer
initial_learning_rate = 1e-3
batch_size = 100
epochs = 200
weight_decay = 0.0005

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


### custom loss

def zero_loss(y_true, y_pred):
    return 0.5 / K.sum(y_pred, axis=0)

def main_block(x, filters, n, strides, dropout):

    # Normal part
    x_res = Conv2D(filters, (3, 3), strides=strides, padding="same",kernel_initializer='lecun_normal')(x)  # , kernel_regularizer=l2(5e-4)
    # x_res = BatchNormalization()(x_res)
    x_res = AlphaDropout(dropout)(x_res)
    x_res = Activation('selu')(x_res)

    x_res = Conv2D(filters, (3, 3), padding="same",kernel_initializer='lecun_normal')(x_res)
    # Alternative branch
    x = Conv2D(filters, (1, 1), strides=strides,kernel_initializer='lecun_normal')(x)
    # Merge Branches
    x = Add()([x_res, x])

    for i in range(n - 1):
        # Residual conection
        # x_res = BatchNormalization()(x)
        x_res = AlphaDropout(dropout)(x_res)
        x_res = Activation('selu')(x_res)

        # x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same",kernel_initializer='lecun_normal')(x_res)
        # Apply dropout if given
        # Second part
        # x_res = BatchNormalization()(x_res)
        x_res = AlphaDropout(dropout)(x_res)
        x_res = Activation('selu')(x_res)

        x_res = Conv2D(filters, (3, 3), padding="same",kernel_initializer='lecun_normal')(x_res)
        # Merge branches
        x = Add()([x, x_res])

    # Inter block part
    # x = BatchNormalization()(x)
    x = AlphaDropout(dropout)(x)
    x = Activation('selu')(x)


    # # Normal part
    # x_res = Conv2D(filters, (3, 3), strides=strides, padding="same")(x)  # , kernel_regularizer=l2(5e-4)
    # x_res = BatchNormalization()(x_res)
    # x_res = Activation('relu')(x_res)
    # x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
    # # Alternative branch
    # x = Conv2D(filters, (1, 1), strides=strides)(x)
    # # Merge Branches
    # x = Add()([x_res, x])
    #
    # for i in range(n - 1):
    #     # Residual conection
    #     x_res = BatchNormalization()(x)
    #     x_res = Activation('relu')(x_res)
    #     x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
    #     # Apply dropout if given
    #     if dropout: x_res = Dropout(dropout)(x)
    #     # Second part
    #     x_res = BatchNormalization()(x_res)
    #     x_res = Activation('relu')(x_res)
    #     x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
    #     # Merge branches
    #     x = Add()([x, x_res])
    #
    # # Inter block part
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    return x


def build_model(inputs,labels, n, k, act="relu", dropout=None):
    """ Builds the model. Params:
            - n: number of layers. WRNs are of the form WRN-N-K
                 It must satisfy that (N-4)%6 = 0
            - k: Widening factor. WRNs are of the form WRN-N-K
                 It must satisfy that K%2 = 0
            - input_dims: input dimensions for the model
            - output_dim: output dimensions for the model
            - dropout: dropout rate - default=0 (not recomended >0.3)
            - act: activation function - default=relu. Build your custom
                   one with keras.backend (ex: swish, e-swish)
    """
    # Ensure n & k are correct
    assert (n - 4) % 6 == 0
    assert k % 2 == 0
    n = (n - 4) // 6

    # Head of the model
    x = Conv2D(64, (3, 3), padding="same",kernel_initializer='lecun_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = keras.layers.PReLU()(x)
    # 3 Blocks (normal-residual)
    x = main_block(x, 16 * k, n, (1, 1), dropout)  # 0
    x = main_block(x, 32 * k, n, (2, 2), dropout)  # 1
    x = main_block(x, 64 * k, n, (2, 2), dropout)  # 2

    # Final part of the model
    x = AveragePooling2D((7,7))(x)
    # x = GlobalAveragePooling3D()(x)
    # print(x.shape)
    x = Flatten()(x)

    side=Dense(2)(x)
    side = PReLU()(side)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([side, labels])
    main = Dense(10, activation='softmax', name='main_out', kernel_regularizer=l2(weight_decay))(x)

    return main, side

def train():
    mnist = read_data_sets('./data/fashion', reshape=False, validation_size=0,
                           source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    main_input = Input((28, 28, 1))
    aux_input = Input((10,))

    final_output, side_output = build_model(inputs=main_input,labels=aux_input, n=16, k=8,dropout=0.2)
    model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])
    model.summary()

    # optim = optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    optim = optimizers.Adam(lr=3e-4)
    model.compile(optimizer=optim,
                  loss={'main_out':losses.categorical_crossentropy,'centerlosslayer':zero_loss},loss_weights=[1,0.01], metrics=['accuracy'])


    train_data = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    # preprocessing_function=random_crop_image,
                                    preprocessing_function=get_random_eraser(v_l=0, v_h=1),
                                    rotation_range=10,
                                    width_shift_range=5. / 28,
                                    height_shift_range=5. / 28,
                                    horizontal_flip=True)
    validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    for data in (train_data, validation_data):
        data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う……。

    best_weights_filepath = './model/best_weights.hdf5'
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,
                                               save_best_only=True, mode='auto')

    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), #当标准评估停止提升时，降低学习速率。
                                   cooldown=0, patience=10, min_lr=1e-6)


    dummy = np.zeros((batch_size, 1))
    def gen_flow_for_two_inputs(gen,X1, y, dummy):
        genX1 = gen.flow(X1, y, batch_size=batch_size, seed=666)
        # genX2 = gen.flow(X1, y1, batch_size=batch_size, seed=666)
        while True:
            X1i = genX1.__next__()
            # X2i = genX2.__next__()
            yield [X1i[0], X1i[1]], [X1i[1],dummy]

    hist = model.fit_generator(gen_flow_for_two_inputs(train_data, x_train, y_train,dummy),
                               # batch_size=batch_size,
                               steps_per_epoch=int(50000 / batch_size),
                               epochs=epochs,
                               verbose=1,
                               validation_data=gen_flow_for_two_inputs(validation_data,x_test, y_test,dummy),
                               validation_steps=int(10000 / batch_size),
                               callbacks=[earlyStopping,saveBestModel,lr_reducer]
                               # validation_data=([x_test, y_test_onehot], [y_test_onehot, dummy2])
                               )



if __name__ == "__main__":
    train()
    # model = build_model((28,28,1), 10, 16, 8)
    # model.summary()