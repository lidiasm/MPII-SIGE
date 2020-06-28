#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo GAN para generar imágenes de los dígitos del conjunto MNIST.
Dispone de un generador con una capa de entrada de 784 neuronas para recrear
las imágenes, una capa oculta de 128 neuronas con función de activación ReLU
y una capa de salida que devuelve la imagen sintética.
Mientras que el discriminador dispone de una única salida que es la probabilidad
de que la imagen sea real o falsa.
"""

# %% Librerías
# Compatibilidad con tensorflow 1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
# Obtener el conjunto de dígitos de MNIST
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# %% Función que genera números aleatorios para 
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# %% Función para dibujar las imágenes generadas
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

# %% Función para definir el generador
def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(z, 128, activation=tf.nn.relu)
        x = tf.layers.dense(z, 784)
        x = tf.nn.sigmoid(x)
    return x

# %% Función para definir el discriminador
def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 1)
        x = tf.nn.sigmoid(x)
    return x

# %% Main
if __name__ == '__main__':
    # Generamos la primera imagen aleatoria
    X = tf.placeholder(tf.float64, shape=[None, 784])
    Z = tf.placeholder(tf.float64, shape=[None, 100])
    # Generador
    G_sample = generator(Z)
    # Discriminador 
    D_real = discriminator(X)
    D_fake = discriminator(G_sample)
    # Funciones de coste
    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))
    # Parámetros de ambos modelos    
    disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]
    # Optimizador Adam para ambos modelos
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=disc_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=gen_vars)
    # Tamaño del lote
    mb_size = 128
    # Dimensión del ruido para generar las primeras imágenes
    Z_dim = 100
    # Descargamos el dataset de dígitos MNIST
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Carpeta para almacenar las imágenes generadas
    if not os.path.exists('imgsGAN/'):
        os.makedirs('imgsGAN/')
        
    # ENTRENAMIENTO
    i = 0
    max_iters = 1000000
    for it in range(max_iters):
        # Guardamos las imágenes generadas cada 1.000 iteraciones en la carpeta
        # creada anteriormente
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
            fig = plot(samples)
            plt.savefig('imgsGAN/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
            
        # Siguiente lote de la imagen
        X_mb, _ = mnist.train.next_batch(mb_size)        
        # Discriminador
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        # Generador
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
