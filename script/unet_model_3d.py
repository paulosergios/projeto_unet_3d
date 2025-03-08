from keras.layers import Input, Conv3D, UpSampling3D, concatenate, BatchNormalization, Dropout, Conv3DTranspose
from keras.models import Model
from keras.regularizers import l2

def build_unet_3d(input_size=(256, 256, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv3D(128, 3, activation='relu', padding='same', strides=(2, 2, 2), kernel_regularizer=l2(0.001))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv3D(256, 3, activation='relu', padding='same', strides=(2, 2, 2), kernel_regularizer=l2(0.001))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv3D(512, 3, activation='relu', padding='same', strides=(2, 2, 2), kernel_regularizer=l2(0.001))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv4)
    conv4 = BatchNormalization()(conv4)

    # Decoder
    up1 = Conv3DTranspose(256, kernel_size=2, strides=(2, 2, 2), padding='same')(conv4)
    up1 = concatenate([conv3, up1], axis=-1)
    conv5 = Conv3D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(up1)
    conv5 = BatchNormalization()(conv5)

    up2 = Conv3DTranspose(128, kernel_size=2, strides=(2, 2, 2), padding='same')(conv5)
    up2 = concatenate([conv2, up2], axis=-1)
    conv6 = Conv3D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(up2)
    conv6 = BatchNormalization()(conv6)

    up3 = Conv3DTranspose(64, kernel_size=2, strides=(2, 2, 2), padding='same')(conv6)
    up3 = concatenate([conv1, up3], axis=-1)
    conv7 = Conv3D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(up3)
    conv7 = BatchNormalization()(conv7)

    outputs = Conv3D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
