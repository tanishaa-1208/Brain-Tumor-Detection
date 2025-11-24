# import numpy as np
# import tensorflow as tf

# def generate_gradcam(model, img_array, last_layer="conv2d_3"):
#     # model must be loaded already (Keras)
#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_layer).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = np.squeeze(heatmap)
#     heatmap = np.maximum(heatmap, 0)
#     if heatmap.max() != 0:
#         heatmap = heatmap / heatmap.max()
#     return heatmap
import numpy as np
import tensorflow as tf

def generate_gradcam(model, img_array, last_layer_name=None):
    """
    Generates a Grad-CAM heatmap for a given model and image array.
    """

    # If user did not specify last layer → automatically find last Conv2D layer
    if last_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_layer_name = layer.name
                break

    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_layer_name).output,
            model.output
        ]
    )

    # Forward + Gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]     # binary classification → index 0

    # Gradients of loss w.r.t. feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Average gradients for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    return heatmap.numpy()
