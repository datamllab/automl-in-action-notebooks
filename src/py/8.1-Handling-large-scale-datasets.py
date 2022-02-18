"""shell
!pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
## 8.1.1 Loading image classification dataset
"""

"""shell
!wget https://github.com/datamllab/automl-in-action-notebooks/raw/master/data/mnist.tar.gz
!tar xzf mnist.tar.gz
"""

"""
```
train/
  0/
    1.png
    21.png
    ...
  1/
  2/
  3/
  ...

test/
  0/
  1/
  ...
```
"""

import os
import autokeras as ak

batch_size = 32
img_height = 28
img_width = 28

parent_dir = "data"

test_data = ak.image_dataset_from_directory(
    os.path.join(parent_dir, "test"),
    seed=123,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
for images, labels in test_data.take(1):
    print(images.shape, images.dtype)
    print(labels.shape, labels.dtype)

"""
## 8.1.2 Splitting the loaded dataset
"""

all_train_data = ak.image_dataset_from_directory(
    os.path.join(parent_dir, "train"),
    seed=123,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
train_data = all_train_data.take(int(60000 / batch_size * 0.8))
validation_data = all_train_data.skip(int(60000 / batch_size * 0.8))

train_data = ak.image_dataset_from_directory(
    os.path.join(parent_dir, "train"),
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

validation_data = ak.image_dataset_from_directory(
    os.path.join(parent_dir, "train"),
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

import tensorflow as tf

train_data = train_data.prefetch(5)
validation_data = validation_data.prefetch(5)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

"""
Then we just do one quick demo of AutoKeras to make sure the dataset works.

"""

clf = ak.ImageClassifier(overwrite=True, max_trials=1)
clf.fit(train_data, epochs=1, validation_data=validation_data)
print(clf.evaluate(test_data))

"""
## 8.1.3 Loading text classification dataset
You can also load text datasets in the same way.

"""

"""shell
!wget https://github.com/datamllab/automl-in-action-notebooks/raw/master/data/imdb.tar.gz
!tar xzf imdb.tar.gz
"""

"""
For this dataset, the data is already split into train and test.
We just load them separately.

"""

import os
import autokeras as ak
import tensorflow as tf

train_data = ak.text_dataset_from_directory(
    "imdb/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    max_length=1000,
    batch_size=32,
).prefetch(1000)

validation_data = ak.text_dataset_from_directory(
    "imdb/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    max_length=1000,
    batch_size=32,
).prefetch(1000)

test_data = ak.text_dataset_from_directory(
    "imdb/test",
    max_length=1000,
).prefetch(1000)

clf = ak.TextClassifier(overwrite=True, max_trials=1)
clf.fit(train_data, epochs=2, validation_data=validation_data)
print(clf.evaluate(test_data))

"""
## 8.1.4 Handling large dataset in general format
"""

data = [5, 8, 9, 3, 6]


def generator():
    for i in data:
        yield i


for x in generator():
    print(x)

dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int32)
for x in dataset:
    print(x.numpy())

import numpy as np

parent_dir = "imdb"


def load_data(path):
    data = []
    for class_label in ["pos", "neg"]:
        for file_name in os.listdir(os.path.join(path, class_label)):
            data.append((os.path.join(path, class_label, file_name), class_label))

    data = np.array(data)
    np.random.shuffle(data)
    return data


def get_generator(data):
    def data_generator():
        for file_path, class_label in data:
            text_file = open(file_path, "r")
            text = text_file.read()
            text_file.close()
            yield text, class_label

    return data_generator


all_train_np = load_data(os.path.join(parent_dir, "train"))


def np_to_dataset(data_np):
    return (
        tf.data.Dataset.from_generator(
            get_generator(data_np),
            output_types=tf.string,
            output_shapes=tf.TensorShape([2]),
        )
        .map(lambda x: (x[0], x[1]))
        .batch(32)
        .prefetch(5)
    )


train_data = np_to_dataset(all_train_np[:20000])
validation_data = np_to_dataset(all_train_np[20000:])
test_np = load_data(os.path.join(parent_dir, "test"))
test_data = np_to_dataset(test_np)

for texts, labels in train_data.take(1):
    print(texts.shape)
    print(labels.shape)

clf = ak.TextClassifier(overwrite=True, max_trials=1)
clf.fit(train_data, epochs=2, validation_data=validation_data)
print(clf.evaluate(test_data))
