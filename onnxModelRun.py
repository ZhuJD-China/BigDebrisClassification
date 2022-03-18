import onnx
import pandas as pd
from pathlib import Path
import tensorflow as tf
import onnxruntime as rt

# Load the onnx model with onnx.load
onnx_model = onnx.load("./models/DenseNet201_20220318_01.onnx")
onnx.checker.check_model(onnx_model)


# Create inference session using ort.InferenceSession
def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("\\")[-2] for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    return df


dir_ = Path('XieTouDataSet_Bin/test')
test_filepaths = list(dir_.glob(r'**/*.jpg'))
print(len(test_filepaths))
test_df = proc_img(test_filepaths)

print(f'Number of pictures in the test dataset: {test_df.shape[0]}\n')

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
# test_images
type(test_images[0][0])
print(test_images[0][0].shape)
"""
test_images[0][0].shape
Out[57]: (28, 224, 224, 3)
"""
# TODO
