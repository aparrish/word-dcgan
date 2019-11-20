# from https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
from __future__ import print_function, division

#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import datetime
import random
import pronouncing as pr
pr.init_cmu()
from PIL import ImageFont, ImageDraw, Image

def gen_word_image(s, font, w, h, xoff):
    ascent, descent = font.getmetrics()
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((xoff, (h/2) - ascent + descent), s, fill=0, font=font)
    return image

def gen_word_batch(size, font, vocab, probs, w=256, h=64, xoff=4):
    words = np.random.choice(vocab, p=probs, size=size)
    bitmaps = np.array(
                [np.array(gen_word_image(s, font, w, h, xoff)) for s in words],
                dtype=np.float32)
    bitmaps = (bitmaps / 127.5) - 1.
    return np.expand_dims(bitmaps, axis=3)

def load_vocab_probs():
    words = []
    probs = []
    with open("cmudict-word-prob.tsv") as fh:
        for line in fh:
            line = line.strip()
            w, p = line.split("\t")
            words.append(w)
            probs.append(float(p))
    probs_arr = np.exp(np.array(probs))
    probs_arr /= probs_arr.sum()
    return words, probs_arr

import matplotlib.pyplot as plt
import sys
import numpy as np

class WordDCGAN():
    def __init__(self, width, height, latent_dim=64):
        # Input shape
        self.img_rows = height
        self.img_cols = width
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.words, self.word_probs = load_vocab_probs()

    def build_generator(self):

        model = Sequential()
        x_val = int(self.img_cols / 4)
        y_val = int(self.img_rows / 4)

        model.add(Dense(128 * y_val * x_val, activation="relu",
            input_dim=self.latent_dim))
        model.add(Reshape((y_val, x_val, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, font, batch_size=128, save_interval=50):

        tstamp = datetime.datetime.utcnow().isoformat()[:19]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            imgs = gen_word_batch(batch_size, font, self.words,
                    self.word_probs, self.img_cols, self.img_rows)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%s %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (tstamp, epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print("saving images...")
                self.save_imgs(epoch)
                print("done")
                print("saving generator model...")
                self.generator.save(
                        "models/%s-%05d-generator.h5" % (tstamp, epoch))
                print("done")

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/word_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = WordDCGAN(64, 16)
    font = ImageFont.truetype('NotoSans-Regular.ttf', size=10)
    dcgan.train(epochs=4000, font=font, batch_size=128, save_interval=1)
