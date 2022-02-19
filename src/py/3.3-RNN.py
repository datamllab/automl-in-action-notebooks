"""shell
pip install -r https://raw.githubusercontent.com/datamllab/automl-in-action-notebooks/master/requirements.txt
"""

"""
# RNN
In this section, we will introduce how to use recurrent neural networks for text
classification.
The dataset we use is the IMDB Movie Reviews.
We use the reviews written by the users as the input and try to predict whether the they
are positive or negative.

## Preparing the data

You can use the following code to load the IMDB dataset.

"""

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_words = 10000
embedding_dim = 32

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=max_words
)
print(train_data.shape)
print(train_labels.shape)
print(train_data[:2])
print(train_labels[:2])

"""
The code above would load the reviews into train_data and test_data, load the labels
(positive or negative) into train_labels and test_labels. As you can see the reviews in
train_data are lists of integers instead of texts. It is because the raw texts cannot be
used as an input to a neural network. Neural networks only accepts numerical data as
inputs.

The integers we see above is the raw text data after a preprocessing step named
tokenization. It first split each review into a list of words and assign an integer to
each of the words. For example, a scentence "How are you? How are you doing?" will be
transformed into a list of words as ["how", "are", "you", "how", "are", "you", "doing"].
Then transformed to [5, 8, 9, 5, 8, 9, 7]. The integers doesn't have special meanings but
a representation of the words. Same integers represents the same words, different
integers represents different words.

The labels are also integers, where 1 represents positive, 0 represents negative.

Then, we pad the data to the same length.
"""

# Pad the sequence to length max_len.
maxlen = 100
print(len(train_data[0]))
print(len(train_data[1]))
train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)
print(train_data.shape)
print(train_labels.shape)

"""
## Building your network
The next step is to build your neural network model and train it.
We will introduce the neural network in three steps.
The first step is the embedding, which transform each integer list into a list of vectors.
The second step is to feed the vectors to the recurrent neural network.
The third step is to use the output of the recurrent neural network for classification.
### Embedding
Embedding means find a corresponding numerical vector for each word, which is now an
integer in the list.
The numerical vector can be seen as the coordinate of a point in the space.
We embed the words into specific points in the space.
That is why we call the process embedding.

To implement it, we use a Keras Embedding layer.
First, we need to create a Keras Sequential model.
Then, we add the layers one by one to the model.
The order of the layers is from the input to the output.
"""

from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential

max_words = 10000
embedding_dim = 32

model = Sequential()
model.add(Embedding(max_words, embedding_dim))
model.summary()

"""
In the code above, we initialized an Embedding layer.
The max_words is the vocabulary size, which is an integer meaning how many different
words are there in the input data.
The integer 16 means the length of the vector representation fro each word is 32.
The output tensor of the Embedding layer is (batch_size, max_len, embedding_dim).
### Recurrent Neural Networks
After the embedding layer, we need to use a recurrent neural network for the
classification.
Recurrent neural networks can handle sequential inputs.
For example, we input a movie review as a sequence of word embedding vectors to it, which
are the output of the embedding layer.
Each vector has a length of 32.
Each review contains 100 vectors.
If we see the RNN as a whole, it takes 100 vectors of length 16 altogether.
However, in the real case, it takes one vector at a time.

Each time the RNN takes in a word embedding vector,
it not only takes the word embedding vector, but another state vector as well.
You can think the state vector as the memory of the RNN.
It memorizes the previous words it taken as input.
In the first step, the RNN has no previous words to remember.
It takes an initial state, which is usually empty,
and the first word embedding vector as input.
The output of the first step is actually the state to be input to the second step.
For the rest of the steps, the RNN will just take the previous output and the current
input as input,
and output the state for the next step.
For the last step, the output state is the final output we will use for the
classification.

We can use the following python code to illustrate the process.
```python
state = [0] * 32
for i in range(100):
    state = rnn(embedding[i], state)
return state
```
The returned state is the final output of the RNN.

Sometimes, we may also need to collect the output of each step as shown in the following
code.
```python
state = [0] * 32
output = []
for i in range(100):
    state = rnn(embedding[i], state)
    output.append(state)
return output
```
In the code above, the output of an RNN can also be a sequence of vectors, which is the
same format as the input to the RNN.
Therefore, we can make the RNN deeper by stacking multiple RNN layers together.


To implement the RNN described, we need the SimpleRNN layer in Keras.
"""

from tensorflow.keras.layers import SimpleRNN

model.add(SimpleRNN(embedding_dim, return_sequences=True))
model.add(SimpleRNN(embedding_dim, return_sequences=True))
model.add(SimpleRNN(embedding_dim, return_sequences=True))
model.add(SimpleRNN(embedding_dim))
model.summary()

"""
The return_sequences parameter controlls whether to collect all the output vectors of an
RNN or only collect the last output. It is set to False by default.
### Classification Head
Then we will use the output of the last SimpleRNN layer, which is a vector of length 32,
as the input to the classification head.
In the classification head, we use a fully-connected layer for the classification.
Then we compile and train the model.

"""

from tensorflow.keras.layers import Dense

model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", metrics=["acc"], loss="binary_crossentropy")
model.fit(train_data, train_labels, epochs=2, batch_size=128)

"""
Then we can validate our model on the testing data.
"""

model.evaluate(test_data, test_labels)
