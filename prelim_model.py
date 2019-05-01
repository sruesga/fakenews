import tensorflow as tf
import json
import numpy as np

class Model():
    def __init__(self, vocab_size=20000, rnn_size=64, learning_rate=1e-4):
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.learning_rate = learning_rate
        self.build_model()
    
    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.rnn_size),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    def train_json(self, json_path):
        print("Loading data...")
        raw_data = json.load(open(json_path))
        ids = sorted(list(raw_data['id'].keys()))
        numerized, labels = raw_data['numerized'], raw_data['type']
        train_data = np.array([numerized[id] for id in ids])
        train_labels = np.array([1 if labels[id] == 'fake' else 0 for id in ids])

        print("Fitting model...")
        history = self.model.fit(train_data, train_labels, epochs=1,
                    validation_split=0.2)

if __name__ == '__main__':
    model = Model()
    model.train_json('fake_reliable_news_headlines.json')

    