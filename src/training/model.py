from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD

from config import LEARNING_RATE


def model(input_shape, num_classes, learning_rate=None, optimizer_name='adam'):
    if learning_rate is None:
        learning_rate = LEARNING_RATE

    model = Sequential()

    # Input
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        input_shape=input_shape
    ))

    # Conv Block 2
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))
    model.add(BatchNormalization())

    # Conv Block 3
    model.add(Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation='relu',
        padding='same'
    ))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # Conv Block 4
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))
    model.add(Dropout(0.5))

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv 1x1
    model.add(Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation='relu',
        padding='same'
    ))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(
        units=32,
        activation='relu'
    ))
    model.add(Dense(
        units=num_classes,
        activation='softmax'
    ))

    optimizer = Adam(learning_rate=learning_rate) if optimizer_name.lower() == 'adam' else SGD(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model.summary()
    return model

