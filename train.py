import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from IPython.display import Markdown, display


def printmd(string):
    # Print with Markdowns
    display(Markdown(string))


# Create a list with the filepaths for training and testing
dir_ = Path('XieTouDataSet_Bin/train')
train_filepaths = list(dir_.glob(r'**/*.jpg'))
print(len(train_filepaths))
dir_ = Path('XieTouDataSet_Bin/test')
test_filepaths = list(dir_.glob(r'**/*.jpg'))
print(len(test_filepaths))


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


# str(train_filepaths[0]).split("\\")[-2]

train_df = proc_img(train_filepaths)
test_df = proc_img(test_filepaths)

print(f'Number of pictures in the training dataset: {train_df.shape[0]}\n')
print(f'Number of pictures in the test dataset: {test_df.shape[0]}\n')
print(f'Number of different labels: {len(train_df.Label.unique())}\n')
print(f'Labels: {train_df.Label.unique()}')

# The DataFrame with the filepaths in one column and the labels in the other one
train_df.head(5)

# Display the number of pictures of each category
vc = train_df['Label'].value_counts()
plt.figure(figsize=(9, 5))
sns.barplot(x=vc.index, y=vc, palette="rocket")
plt.title("Number of pictures of each category", fontsize=15)
plt.show()

# Display some pictures of the dataset
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 7),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(train_df.Filepath[i]))
    ax.set_title(train_df.Label[i], fontsize=15)
plt.tight_layout(pad=0.5)
plt.show()


def create_gen():
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
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

    return train_generator, test_generator, train_images, val_images, test_images


def get_model(model):
    # Load the pretained model
    kwargs = {'input_shape': (224, 224, 3),
              'include_top': False,
              'weights': 'imagenet',
              'pooling': 'avg'}

    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False

    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Dictionary with the models
models = {
    "DenseNet121": {"model": tf.keras.applications.DenseNet121, "perf": 0},
    "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
    "DenseNet169": {"model": tf.keras.applications.DenseNet169, "perf": 0},
    "DenseNet201": {"model": tf.keras.applications.DenseNet201, "perf": 0},
    "EfficientNetB0": {"model": tf.keras.applications.EfficientNetB0, "perf": 0},
    "EfficientNetB1": {"model": tf.keras.applications.EfficientNetB1, "perf": 0},
    "EfficientNetB2": {"model": tf.keras.applications.EfficientNetB2, "perf": 0},
    "EfficientNetB3": {"model": tf.keras.applications.EfficientNetB3, "perf": 0},
    "EfficientNetB4": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
    "EfficientNetB5": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
    "EfficientNetB6": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
    "EfficientNetB7": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
    "InceptionResNetV2": {"model": tf.keras.applications.InceptionResNetV2, "perf": 0},
    "InceptionV3": {"model": tf.keras.applications.InceptionV3, "perf": 0},
    "MobileNet": {"model": tf.keras.applications.MobileNet, "perf": 0},
    "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
    "MobileNetV3Large": {"model": tf.keras.applications.MobileNetV3Large, "perf": 0},
    "MobileNetV3Small": {"model": tf.keras.applications.MobileNetV3Small, "perf": 0},
    #     "NASNetLarge": {"model":tf.keras.applications.NASNetLarge, "perf":0}, Deleted because the input shape has to be another one
    "NASNetMobile": {"model": tf.keras.applications.NASNetMobile, "perf": 0},
    "ResNet101": {"model": tf.keras.applications.ResNet101, "perf": 0},
    "ResNet101V2": {"model": tf.keras.applications.ResNet101V2, "perf": 0},
    "ResNet152": {"model": tf.keras.applications.ResNet152, "perf": 0},
    "ResNet152V2": {"model": tf.keras.applications.ResNet152V2, "perf": 0},
    "ResNet50": {"model": tf.keras.applications.ResNet50, "perf": 0},
    "ResNet50V2": {"model": tf.keras.applications.ResNet50V2, "perf": 0},
    "VGG16": {"model": tf.keras.applications.VGG16, "perf": 0},
    "VGG19": {"model": tf.keras.applications.VGG19, "perf": 0},
    "Xception": {"model": tf.keras.applications.Xception, "perf": 0}
}

# Create the generators
train_generator, test_generator, train_images, val_images, test_images = create_gen()
print('\n')

# Fit the models
for name, model in models.items():
    # Get the model
    m = get_model(model['model'])
    models[name]['model'] = m

    start = perf_counter()

    # Fit the model
    history = m.fit(train_images, validation_data=val_images, epochs=1, verbose=0)

    # Sav the duration and the val_accuracy
    duration = perf_counter() - start
    duration = round(duration, 2)
    models[name]['perf'] = duration
    print(f"{name:20} trained in {duration} sec")

    val_acc = history.history['val_accuracy']
    models[name]['val_acc'] = [round(v, 4) for v in val_acc]

for name, model in models.items():
    # Predict the label of the test_images
    pred = models[name]['model'].predict(test_images)
    pred = np.argmax(pred, axis=1)

    # Map the label
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    y_test = list(test_df.Label)
    acc = accuracy_score(y_test, pred)
    models[name]['acc'] = round(acc, 4)

# Create a DataFrame with the results
models_result = []

for name, v in models.items():
    models_result.append([name, models[name]['val_acc'][-1],
                          models[name]['acc'],
                          models[name]['perf']])

df_results = pd.DataFrame(models_result,
                          columns=['model', 'val_accuracy', 'accuracy', 'Training time (sec)'])
df_results.sort_values(by='accuracy', ascending=False, inplace=True)
df_results.reset_index(inplace=True, drop=True)
df_results

plt.figure(figsize=(15, 5))
sns.barplot(x='model', y='val_accuracy', data=df_results)
plt.title('val_accuracy on the test set (after 1 epoch))', fontsize=15)
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.show()

# Create and train the model
model = get_model(tf.keras.applications.DenseNet201)
history = model.fit(train_images,
                    validation_data=val_images,
                    epochs=50,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True)]
                    )
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title("Loss")
plt.show()

# Predict the label of the test_images
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred = [labels[k] for k in pred]

# Get the accuracy on the test set
y_test = list(test_df.Label)

acc = accuracy_score(y_test, pred)
print(f'# Accuracy on the test set: {acc * 100:.2f}%')

# Display a confusion matrix
from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=True, xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)), cbar=False)
plt.title('Normalized Confusion Matrix', fontsize=23)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Display picture of the dataset with their labels
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i].split('_')[0]}\nPredicted: {pred[i].split('_')[0]}", fontsize=15)
plt.tight_layout()
plt.show()
