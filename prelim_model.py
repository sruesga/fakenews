import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt

class Model():
    def load_json(self, json_path):
        print("Loading data...")
        raw_data = json.load(open(json_path))
        ids = sorted(list(raw_data['id'].keys()))
        numerized, labels = raw_data['numerized'], raw_data['type']
        self.train_data = np.array([numerized[id] for id in ids])
        self.train_labels = np.array([1 if labels[id] == 'fake' else 0 for id in ids])

    def train(self):
        print("Fitting model...")
        self.history = self.model.fit(self.train_data, self.train_labels, epochs=10,
                    validation_split=0.2)#, callbacks=[NBatchLogger(display=100)])
        self.save_model()
    
    def plot_acc(self, title, model_name):
        h = self.history.history
        plt.plot(h['val_acc'], label=model_name + ' valid')
        plt.plot(h['acc'], label=model_name + ' train')
        plt.xlabel('epoch')
        plt.title(title)
        plt.legend()
        plt.show()
    
    def plot_loss(self, title, model_name):
        h = self.history.history
        plt.plot(h['val_loss'], label=model_name + ' valid')
        plt.plot(h['loss'], label=model_name + ' train')
        plt.xlabel('epoch')
        plt.title(title)
        plt.legend()
        plt.show()

    
    def save_model(self):

        self.model.save_weights(self.weights_path)
    
    def load_model(self):
        self.model.load_weights(self.weights_path)


class RNNModel(Model):
    def __init__(self, vocab_size=20000, rnn_size=64, weights_path='lstm_weights.h5', learning_rate=1e-4, model_choice='lstm', epochs=10):
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.learning_rate = learning_rate
        self.weights_path = weights_path
        self.epochs = epochs
        self.build_model(model_choice)
    
    def build_model(self, model_choice):
        if model_choice == 'lstm':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.rnn_size),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        elif model_choice == 'rnn':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.rnn_size),
                tf.keras.layers.SimpleRNN(self.rnn_size),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

class ConvModel(Model):
    def __init__(self, vocab_size=20000, embed_size=64, num_filt=32, weights_path='conv_weights.h5', learning_rate=1e-4, model_choice='1dconv'):
        self.vocab_size = vocab_size
        self.num_filt = num_filt
        self.learning_rate = learning_rate
        self.weights_path = weights_path
        self.embed_size = embed_size
        self.build_model(model_choice)

    def build_model(self, model_choice):
        if model_choice == '1dconv':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.embed_size),
                tf.keras.layers.Conv1D(self.num_filt, 4, strides=2, padding="same"),
                tf.keras.layers.Conv1D(self.num_filt, 4, strides=2),
                tf.keras.layers.Conv1D(1, 4),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])


if __name__ == '__main__':
    model = ConvModel(model_choice='1dconv', weights_path='cnn_e-4.h5')
    model.load_json('fake_reliable_news_headlines.json')
    model.train()

    