# from https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
from __future__ import print_function, division

#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Embedding, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import datetime
import random
import json

# text transformation functions

def ucfirst(s):
    return s[0].upper() + s[1:]

def allcaps(s):
    return s.upper()

def randpunct(s):
    probs = {'!': 0.1746109907425645,
             '.': 0.7268071695883396,
             '?': 0.09858183966909592}
    return s + np.random.choice(list(probs.keys()), p=list(probs.values()))

class Style:
    "a 'style' wraps a text transformation technique with a font"
    def __init__(self, desc, transform, font, size, label):
        self.desc = desc
        self.transform = transform
        self.font = font
        self.label = label
        self.size = size

def gen_word_image(s, w, h, xoff, style):
    "generate a word image for the given string using given style and metrics"
    ascent, descent = style.font.getmetrics()
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((xoff, (h/2) - ascent + descent), style.transform(s),
            fill=0, font=style.font)
    return image

def gen_word_batch(size, vocab, probs, w=256, h=64, xoff=4,
        styles=None, dist='unigram'):
    "generate a batch of words with given metrics, sampling from given styles"

    if dist == 'unigram':
        words = np.random.choice(vocab, p=probs, size=size)
    else:
        words = np.random.choice(vocab, size=size)

    style_indices = np.random.randint(0, len(styles), size=size)

    bitmaps = []
    labels = []
    for i, idx in enumerate(style_indices):
        word_img  = np.array(gen_word_image(words[i], w, h, xoff, styles[idx]))
        bitmaps.append(word_img)
        labels.append(styles[idx].label)

    bitmaps = np.array(bitmaps, dtype=np.float32)
    labels = np.array(labels)
    bitmaps = (bitmaps / 127.5) - 1.
    return (np.expand_dims(bitmaps, axis=3), labels)

def build_glyph_width_map(style, alphabet, w=128, h=128):
    "precalculate glyph widths for a given style"
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    widths = {}
    for ch in alphabet:
        widths[ch] = draw.textsize(ch, style.font)[0]
    return widths

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-.'"

def load_vocab_probs(w, h, xoff, styles, debug=False):
    "load probabilities for words, excluding words that won't fit on canvas"
    words = []
    probs = []
    # pillow textsize takes too long, so we'll just sum up the width of the
    # characters as a proxy. sorry ramsey
    glyph_width_maps = [build_glyph_width_map(st, alphabet) for st in styles]
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)
    with open("cmudict-word-prob.tsv") as fh:
        for line in fh:
            skipped = False
            line = line.strip()
            word, p = line.split("\t")
            # if it's too big in *any* style, skip
            for st, wmap in zip(styles, glyph_width_maps):
                if sum(wmap.get(ch, 4) for ch in st.transform(word)) > w * 0.9:
                #if draw.textsize(st.transform(word), st.font)[0]+xoff > w * 0.9:
                    if debug:
                        print("skipping", word, "--- too long in", st.desc)
                    skipped = True
                    break
            if not skipped:
                words.append(word)
                probs.append(float(p))
    probs_arr = np.exp(np.array(probs))
    probs_arr /= probs_arr.sum()
    return words, probs_arr


class WordDCGAN():
    def __init__(self, width, height, words, word_probs, styles, latent_dim=64):

        # Input shape
        self.img_rows = height
        self.img_cols = width
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = latent_dim
        self.class_dim = len(styles)

        self.styles = styles

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # The generator takes noise/classes as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        c = Input(shape=(1,))
        img = self.generator([z, c])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([img, c])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, c], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.words = words
        self.word_probs = word_probs

    def build_generator(self):

        x_val = int(self.img_cols / 4)
        y_val = int(self.img_rows / 4)

        # noise input
        z = Input(shape=(self.latent_dim,))

        # class input
        c = Input(shape=(1,), dtype='int32')
        c_embed = Embedding(self.class_dim, self.latent_dim,
            embeddings_initializer='glorot_uniform')(c)
        c_bnormal = BatchNormalization(momentum=0.8)(c_embed)

        # combo
        zc = multiply([z, c_bnormal])

        dense = Dense(128 * y_val * x_val, activation="relu")(zc)
        reshape = Reshape((y_val, x_val, 128))(dense)
        upsample1 = UpSampling2D()(reshape)
        conv2d1 = Conv2D(128, kernel_size=3, padding="same")(upsample1)
        bnormal1 = BatchNormalization(momentum=0.8)(conv2d1)
        #relu1 = Activation("relu")(bnormal1)
        relu1 = LeakyReLU(alpha=0.2)(bnormal1)
        upsample2 = UpSampling2D()(relu1)
        conv2d2 = Conv2D(64, kernel_size=3, padding="same")(upsample2)
        bnormal2 = BatchNormalization(momentum=0.8)(conv2d2)
        #relu2 = Activation("relu")(bnormal2)
        relu2 = LeakyReLU(alpha=0.2)(bnormal2)
        conv2d3 = Conv2D(self.channels, kernel_size=3, padding="same",
                activation="tanh")(relu2)

        return Model([z, c], conv2d3)

    def build_discriminator(self):

        input_img = Input(self.img_shape)

        # class input
        c = Input(shape=(1,), dtype='int32')
        c_embed = Embedding(self.class_dim,
                (self.img_shape[0]*self.img_shape[1]*self.img_shape[2]),
                embeddings_initializer='glorot_uniform')(c)
        c_embed_reshape = Reshape(self.img_shape)(c_embed)
        #c_bnormal = BatchNormalization(momentum=0.8)(c_embed_reshape)

        zc = multiply([input_img, c_embed_reshape])

        conv2d_img1 = Conv2D(32, kernel_size=3, strides=2,
                input_shape=self.img_shape, padding="same")(zc)
        lrlu_img1 = LeakyReLU(alpha=0.2)(conv2d_img1)
        dropout_img1 = Dropout(0.25)(lrlu_img1)
        conv2d_img2 = Conv2D(64, kernel_size=3, strides=2,
                padding="same")(dropout_img1)
        zeropad_img1 = ZeroPadding2D(padding=((0,1),(0,1)))(conv2d_img2)
        bnormal_img1 = BatchNormalization(momentum=0.8)(zeropad_img1)
        lrlu_img2 = LeakyReLU(alpha=0.2)(bnormal_img1)
        dropout_img2 = Dropout(0.25)(lrlu_img2)
        conv2d_img3 = Conv2D(128, kernel_size=3, strides=2,
                padding="same")(dropout_img2)
        bnormal_img2 = BatchNormalization(momentum=0.8)(conv2d_img3)
        lrlu_img3 = LeakyReLU(alpha=0.2)(bnormal_img2)
        dropout_img3 = Dropout(0.25)(lrlu_img3)
        conv2d_img4 = Conv2D(256, kernel_size=3, strides=1,
                padding="same")(dropout_img3)
        bnormal_img3 = BatchNormalization(momentum=0.8)(conv2d_img4)
        lrlu_img4 = LeakyReLU(alpha=0.2)(bnormal_img3)
        dropout_img4 = Dropout(0.25)(lrlu_img4)
        flatten = Flatten()(dropout_img4)
        dense_out = Dense(1, activation='sigmoid')(flatten)

        return Model([input_img, c], dense_out)

    def train(self, epochs, batch_size=128, save_interval=50, word_dist='unigram'):

        tstamp = datetime.datetime.utcnow().isoformat()[:19]
        print("starting training at", tstamp)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            imgs, classes = gen_word_batch(batch_size, self.words,
                    self.word_probs, self.img_cols, self.img_rows,
                    int(self.img_cols * 0.05), self.styles)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            rand_classes = np.random.randint(0, self.class_dim,
                    size=batch_size)
            gen_imgs = self.generator.predict([noise, rand_classes])

            # Train the discriminator (real as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(
                    [imgs, classes], valid)
            d_loss_fake = self.discriminator.train_on_batch(
                    [gen_imgs, rand_classes], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, rand_classes], valid)

            # Plot the progress
            print ("%s %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % \
                    (tstamp, epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == (save_interval - 1):
                print("saving images...")
                self.save_imgs(self.styles, epoch+1, tstamp)
                print("done")
                print("saving generator model...")
                self.generator.save(
                        "models/%s-%05d-generator.h5" % (tstamp, epoch+1))
                print("done")

    def save_imgs(self, styles, epoch, tstamp):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        labels = np.array([st.label for st in np.random.choice(styles,
            size=r*c)])
        gen_imgs = self.generator.predict([noise, labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(12, 8))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s_word_%d.png" % (tstamp, epoch))
        plt.close()


"""

{
    'styles': [
        {
            'font': 'foo.ttf',
            'transform': 'lc',
            'label': 0,
            'size': 12
        }
    ]
}

"""
def readstyleconf(fh):
    data = json.load(fh)
    styles_def = data['styles']
    transforms = {
        'ucfirst': ucfirst,
        'allcaps': allcaps,
        'randpunct': randpunct,
        'none': lambda x: x
    }
    styles = []
    for sdict in styles_def:
        style = Style(
            desc=sdict['desc'],
            transform=transforms[sdict['transform']],
            font=ImageFont.truetype(sdict['font'], size=sdict['size']),
            size=sdict['size'],
            label=sdict['label'])
        styles.append(style)
    return styles

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='word dcgan training script!')
    parser.add_argument(
            '--img-width',
            type=int,
            default=128,
            help='width of generated word bitmaps (should be power of 2)')
    parser.add_argument(
            '--img-height',
            type=int,
            default=32,
            help='height of generated word bitmaps (should be power of 2)')
    parser.add_argument(
            '--style-conf',
            type=argparse.FileType('r'),
            help='json file with config for text fonts/sizes/labels')
    parser.add_argument(
            '--word-dist',
            choices=['unigram', 'uniform'],
            default='unigram',
            help='word distribution (unigram=spacy unigram frequency, uniform=uniform probability)')
    parser.add_argument(
            '--epochs',
            type=int,
            default=5000,
            help='number of epochs to train')
    parser.add_argument(
            '--batch-size',
            type=int,
            default=128,
            help='number of samples per epoch')
    parser.add_argument(
            '--save-interval',
            type=int,
            default=50,
            help='save images and model every N epochs')
    parser.add_argument(
            '--latent-dim',
            type=int,
            default=64,
            help='latent dimension count')
    args = parser.parse_args()

    with args.style_conf as fh:
        print("reading styles...")
        styles = readstyleconf(fh)

    print("loading word probabilities...")
    words, word_probs = load_vocab_probs(
            args.img_width,
            args.img_height,
            args.img_width * 0.05,
            styles,
            debug=True)

    print("initializing DCGAN model...")
    dcgan = WordDCGAN(
            args.img_width,
            args.img_height,
            words,
            word_probs,
            styles,
            args.latent_dim)

    print("training DCGAN model...")
    dcgan.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            word_dist=args.word_dist)

