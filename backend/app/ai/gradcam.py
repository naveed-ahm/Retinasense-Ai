import numpy as np
import cv2
import tensorflow as tf


import numpy as np
import cv2
import tensorflow as tf


def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int,
    original_image: np.ndarray | None = None,
    alpha: float = 0.45,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        base = model.get_layer("efficientnetb3")
        top_conv = base.get_layer("top_conv")

        top_conv_idx = base.layers.index(top_conv)
        remaining_base_layers = base.layers[top_conv_idx + 1 :]
        head_layers = model.layers[2:]

        conv_input = tf.keras.Input(shape=top_conv.output.shape[1:])
        x = conv_input
        for layer in remaining_base_layers:
            x = layer(x)
        for layer in head_layers:
            x = layer(x)

        head_model = tf.keras.Model(inputs=conv_input, outputs=x)
        conv_model = tf.keras.Model(inputs=base.inputs, outputs=top_conv.output)

        with tf.GradientTape() as tape:
            conv_feat = conv_model(image, training=False)
            tape.watch(conv_feat)
            preds = head_model(conv_feat, training=False)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_feat)
        if grads is None:
            return None, None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_feat[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        out_shape = (original_image.shape[1], original_image.shape[0]) if original_image is not None else (512, 512)
        heatmap = cv2.resize(heatmap, out_shape)
        heatmap = np.uint8(255 * heatmap)
        jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        if original_image is not None:
            base_img = original_image.copy()
            if base_img.max() <= 1.0:
                base_img = np.uint8(base_img * 255)
            if base_img.shape[:2] != (out_shape[1], out_shape[0]):
                base_img = cv2.resize(base_img, out_shape)
            overlay = cv2.addWeighted(base_img, 1 - alpha, jet, alpha, 0)
        else:
            overlay = jet

        return heatmap, overlay
    except Exception:
        return None, None


def generate_gradcam_batch(
    model: tf.keras.Model,
    images: np.ndarray,
    class_indices: list[int],
    original_images: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
    results = []
    for i in range(len(images)):
        orig = original_images[i] if original_images and i < len(original_images) else None
        heatmap, overlay = generate_gradcam(model, images[i:i+1], class_indices[i], orig)
        results.append((heatmap, overlay))
    return results
