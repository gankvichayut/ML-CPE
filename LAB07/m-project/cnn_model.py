import json
import os
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, num_conv_layers=3, dense_units=128):
    """
    สร้างโมเดล CNN ที่มีเสถียรภาพในการเทรน
    - ลด Learning Rate เพื่อไม่ให้กราฟเหวี่ยง
    - ปรับ Augmentation ให้เหมาะสมกับรูปมือ
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255.0))
    
    # ใช้ Augmentation แบบเบาเพื่อไม่ให้รูปมือเสียทรง
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.05))

    filters = [32, 64, 128, 256]
    for i in range(num_conv_layers):
        f = filters[i] if i < len(filters) else 256
        model.add(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # ลด Learning Rate เป็น 1e-4 เพื่อให้เส้น Loss/Accuracy นิ่งและลู่เข้าแบบสมูท
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                num_conv_layers=3, dense_units=128, epochs=30, batch_size=64):
    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=num_classes,
        num_conv_layers=num_conv_layers,
        dense_units=dense_units
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )
    return model, history


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)