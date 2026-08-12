import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB3
from app.ai.config import ModelConfig


def build_model(config: ModelConfig | None = None, train_base: bool = False) -> Model:
    if config is None:
        config = ModelConfig()
    base_model = EfficientNetB3(
        weights=config.weights,
        include_top=False,
        input_shape=config.input_shape,
        pooling="avg",
    )
    base_model.trainable = train_base
    for layer in base_model.layers:
        layer.trainable = train_base
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    if train_base:
        # Unfreeze only the last N layers (BN stays frozen to protect running stats).
        for layer in base_model.layers[-config.trainable_base_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    inputs = tf.keras.Input(shape=config.input_shape, name="input_image")
    # training flag must match train_base: False while base is frozen (no BN stat updates),
    # True during fine-tuning so BN/conv layers train in proper mode.
    x = base_model(inputs, training=train_base)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout_rate * 0.5)(x)
    x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(config.num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=outputs, name="retinasense")
    return model


def freeze_base(model: Model, freeze: bool = True):
    for layer in model.layers[1].layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = not freeze
