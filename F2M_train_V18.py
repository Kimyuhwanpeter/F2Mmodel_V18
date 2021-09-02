# -*- coding:utf-8 -*-
from F2M_model_V18 import *

from random import shuffle, random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256, 
                           
                           "load_size": 276,

                           "tar_size": 256,

                           "tar_load_size": 276,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "D:/[1]DB/[5]4th_paper_DB/Generation/Morph/train_BM.txt",
                           
                           "A_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/Crop_dlib/",
                           
                           "B_txt_path": "D:/[1]DB/[5]4th_paper_DB/Generation/Morph/train_WM.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/Crop_dlib/",

                           "age_range": [40, 64],

                           "n_classes": 256,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Pictures/img2",
                           
                           "A_test_txt_path": "",
                           
                           "A_test_img_path": "",
                           
                           "B_test_txt_path": "",
                           
                           "B_test_img_path": "",
                           
                           "test_dir": "A2B",
                           
                           "fake_B_path": "",
                           
                           "fake_A_path": ""})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data[0])
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.tar_load_size, FLAGS.tar_load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.tar_size, FLAGS.tar_size, 3])
    B_img = B_img / 127.5 - 1.

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    B_lab = int(B_data[1])
    A_lab = int(A_data[1])

    return A_img, A_lab, B_img, B_lab

def te_input_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    lab = lab

    return img, lab

#@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def decreas_func(x):
    return tf.maximum(0, tf.math.exp(x * (-2.77 / 100)))

def increase_func(x):
    x = tf.cast(tf.maximum(1, x), tf.float32)
    return tf.math.log(x + 1e-7)

def grad_norm(gradients):
    norm = tf.norm(
        tf.stack([tf.norm(grad) for grad in gradients if grad is not None]))
    return norm

def cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
             A_batch_images, B_batch_images, B_batch_labels, A_batch_labels):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_B = model_out(A2B_G_model, A_batch_images, True)
        fake_A_ = model_out(B2A_G_model, tf.nn.tanh(fake_B[:, :, :, 0:3]), True)

        fake_A = model_out(B2A_G_model, B_batch_images, True)
        fake_B_ = model_out(A2B_G_model, tf.nn.tanh(fake_A[:, :, :, 0:3]), True)

        # identification    # ?̰͵? ?߰??ϸ? ?????? ???????
        #id_fake_A, _ = model_out(B2A_G_model, A_batch_images, True)
        #id_fake_B, _ = model_out(A2B_G_model, B_batch_images, True)

        DB_real = model_out(B_discriminator, B_batch_images, True)
        DB_fake = model_out(B_discriminator, tf.nn.tanh(fake_B[:, :, :, 0:3]), True)
        DA_real = model_out(A_discriminator, A_batch_images, True)
        DA_fake = model_out(A_discriminator, tf.nn.tanh(fake_A[:, :, :, 0:3]), True)

        ################################################################################################
        # ???̿? ???? distance?? ???ϴ°?
        return_loss = 0.
        for i in range(FLAGS.batch_size):   # ?????? ?????ͷ??Ϸ??? compare label?? ?ϳ? ?? ???????? ?Ѵ? ??????!!!!
            energy_ft = tf.reduce_sum(tf.abs(tf.reduce_mean(fake_A[i, :, :, 3:], [0,1]) - tf.reduce_mean(fake_B[:, :, :, 3:], [1,2])), 1)
            energy_ft2 = tf.reduce_sum(tf.abs(tf.reduce_mean(fake_A_[i, :, :, 3:], [0,1]) - tf.reduce_mean(fake_B_[:, :, :, 3:], [1,2])), 1)
            compare_label = tf.subtract(A_batch_labels, B_batch_labels[i])

            T = 4
            label_buff = tf.less(tf.abs(compare_label), T)
            label_cast = tf.cast(label_buff, tf.float32)

            realB_fakeB_loss = label_cast * increase_func(energy_ft) \
                + (1 - label_cast) * 1 * decreas_func(energy_ft)

            realA_fakeA_loss = label_cast * increase_func(energy_ft2) \
                + (1 - label_cast) * 1 * decreas_func(energy_ft2)

            # A?? B ???̰? ?ٸ??? ?????Լ?, ?????? ?????Լ?

            loss_buf = 0.
            for j in range(FLAGS.batch_size):
                loss_buf += realB_fakeB_loss[j] + realA_fakeA_loss[j]
            loss_buf /= FLAGS.batch_size

            return_loss += loss_buf
        return_loss /= FLAGS.batch_size
        ################################################################################################
        # content loss ?? ?ۼ?????
        f_B = tf.nn.tanh(fake_B[:, :, :, 0:3])
        f_B_x, f_B_y = tf.image.image_gradients(f_B)
        f_B_m = tf.add(tf.abs(f_B_x), tf.abs(f_B_y))
        f_B = tf.abs(f_B - f_B_m)

        f_A = tf.nn.tanh(fake_A[:, :, :, 0:3])
        f_A_x, f_A_y = tf.image.image_gradients(f_A)
        f_A_m = tf.add(tf.abs(f_A_x), tf.abs(f_A_y))
        f_A = tf.abs(f_A - f_A_m)

        r_A = A_batch_images
        r_A_x, r_A_y = tf.image.image_gradients(r_A)
        r_A_m = tf.add(tf.abs(r_A_x), tf.abs(r_A_y))
        r_A = tf.abs(r_A - r_A_m)

        r_B = B_batch_images
        r_B_x, r_B_y = tf.image.image_gradients(r_B)
        r_B_m = tf.add(tf.abs(r_B_x), tf.abs(r_B_y))
        r_B = tf.abs(r_B - r_B_m)

        id_loss = tf.reduce_mean(tf.abs(f_B - r_A)) * 5.0 \
            + tf.reduce_mean(tf.abs(f_A - r_B)) * 5.0   # content loss  # style?? ?ƴ? skin  ???? ?̶??? ????
        # target?? ?????? ?????°??̱⿡, ?????? ?????? target???? ???? ?? ???? ?????? ???? ?ǵ???

        #Cycle_loss = (tf.reduce_mean(tf.abs(tf.nn.tanh(fake_A_[:, :, :, 0:3]) - A_batch_images))) \
        #    * 10.0 + (tf.reduce_mean(tf.abs(tf.nn.tanh(fake_B_[:, :, :, 0:3]) - B_batch_images))) * 10.0
        # Cycle?? ?Ͽ? ???????? ????, Cycle?? ?̹????? high freguency ?????? ???? ???а? ????????????
        ############################ high freqency ???? ?̿? ############################
        
        f_A_x_, f_A_y = tf.image.image_gradients(tf.nn.tanh(fake_A_[:, :, :, 0:3]))
        f_A_m_ = tf.add(tf.abs(f_A_x_), tf.abs(f_A_y))
        f_A_m_low = tf.abs(tf.nn.tanh(fake_A_[:, :, :, 0:3]) - f_A_m_)
        r_A_m_los = tf.abs(A_batch_images - r_A_m)

        f_B_x_, f_B_y = tf.image.image_gradients(tf.nn.tanh(fake_B_[:, :, :, 0:3]))
        f_B_m_ = tf.add(tf.abs(f_B_x_), tf.abs(f_B_y))
        f_B_m_low = tf.abs(tf.nn.tanh(fake_B_[:, :, :, 0:3]) - f_B_m_)
        r_B_m_los = tf.abs(B_batch_images - r_B_m)

        Cycle_loss = (tf.reduce_mean(tf.abs(f_A_m_ - r_A_m))) \
            * 5.0 + (tf.reduce_mean(tf.abs(f_B_m_ - r_B_m))) * 5.0 \
            + 5.0 + (tf.reduce_mean(tf.abs(f_A_m_low - r_A_m_los))) \
            + 5.0 * (tf.reduce_mean(tf.abs(f_B_m_low - r_B_m_los)))
        ################## high freqency ???? ?̿? ##################       # style ?̶??? ????

        G_gan_loss = tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2) \
            + tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2)

        Adver_loss = (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2. \
            + (tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + return_loss + id_loss
        d_loss = Adver_loss

    g_trainables_params = A2B_G_model.trainable_variables + B2A_G_model.trainable_variables
    d_trainables_params = A_discriminator.trainable_variables + B_discriminator.trainable_variables
    g_grads = g_tape.gradient(g_loss, g_trainables_params)
    d_grads = d_tape.gradient(d_loss, d_trainables_params)

    g_grads_norm = grad_norm(g_grads)
    g_grads_scale = 0.05 / g_grads_norm
    d_grads_norm = grad_norm(d_grads)
    d_grads_scale = 0.05 / d_grads_norm

    ################################################################################################################
    e_w_g_buf = []
    e_w_d_buf = []
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

        fake_B = model_out(A2B_G_model, A_batch_images, True)
        fake_A_ = model_out(B2A_G_model, tf.nn.tanh(fake_B[:, :, :, 0:3]), True)

        fake_A = model_out(B2A_G_model, B_batch_images, True)
        fake_B_ = model_out(A2B_G_model, tf.nn.tanh(fake_A[:, :, :, 0:3]), True)

        # identification    # ?̰͵? ?߰??ϸ? ?????? ???????
        #id_fake_A, _ = model_out(B2A_G_model, A_batch_images, True)
        #id_fake_B, _ = model_out(A2B_G_model, B_batch_images, True)

        DB_real = model_out(B_discriminator, B_batch_images, True)
        DB_fake = model_out(B_discriminator, tf.nn.tanh(fake_B[:, :, :, 0:3]), True)
        DA_real = model_out(A_discriminator, A_batch_images, True)
        DA_fake = model_out(A_discriminator, tf.nn.tanh(fake_A[:, :, :, 0:3]), True)

        ################################################################################################
        # ???̿? ???? distance?? ???ϴ°?
        return_loss = 0.
        for i in range(FLAGS.batch_size):   # ?????? ?????ͷ??Ϸ??? compare label?? ?ϳ? ?? ???????? ?Ѵ? ??????!!!!
            energy_ft = tf.reduce_sum(tf.abs(tf.reduce_mean(fake_A[i, :, :, 3:], [0,1]) - tf.reduce_mean(fake_B[:, :, :, 3:], [1,2])), 1)
            energy_ft2 = tf.reduce_sum(tf.abs(tf.reduce_mean(fake_A_[i, :, :, 3:], [0,1]) - tf.reduce_mean(fake_B_[:, :, :, 3:], [1,2])), 1)
            compare_label = tf.subtract(A_batch_labels, B_batch_labels[i])

            T = 4
            label_buff = tf.less(tf.abs(compare_label), T)
            label_cast = tf.cast(label_buff, tf.float32)

            realB_fakeB_loss = label_cast * increase_func(energy_ft) \
                + (1 - label_cast) * 1 * decreas_func(energy_ft)

            realA_fakeA_loss = label_cast * increase_func(energy_ft2) \
                + (1 - label_cast) * 1 * decreas_func(energy_ft2)

            # A?? B ???̰? ?ٸ??? ?????Լ?, ?????? ?????Լ?

            loss_buf = 0.
            for j in range(FLAGS.batch_size):
                loss_buf += realB_fakeB_loss[j] + realA_fakeA_loss[j]
            loss_buf /= FLAGS.batch_size

            return_loss += loss_buf
        return_loss /= FLAGS.batch_size
        ################################################################################################
        # content loss ?? ?ۼ?????
        f_B = tf.nn.tanh(fake_B[:, :, :, 0:3])
        f_B_x, f_B_y = tf.image.image_gradients(f_B)
        f_B_m = tf.add(tf.abs(f_B_x), tf.abs(f_B_y))
        f_B = tf.abs(f_B - f_B_m)

        f_A = tf.nn.tanh(fake_A[:, :, :, 0:3])
        f_A_x, f_A_y = tf.image.image_gradients(f_A)
        f_A_m = tf.add(tf.abs(f_A_x), tf.abs(f_A_y))
        f_A = tf.abs(f_A - f_A_m)

        r_A = A_batch_images
        r_A_x, r_A_y = tf.image.image_gradients(r_A)
        r_A_m = tf.add(tf.abs(r_A_x), tf.abs(r_A_y))
        r_A = tf.abs(r_A - r_A_m)

        r_B = B_batch_images
        r_B_x, r_B_y = tf.image.image_gradients(r_B)
        r_B_m = tf.add(tf.abs(r_B_x), tf.abs(r_B_y))
        r_B = tf.abs(r_B - r_B_m)

        id_loss = tf.reduce_mean(tf.abs(f_B - r_A)) * 5.0 \
            + tf.reduce_mean(tf.abs(f_A - r_B)) * 5.0   # content loss  # style?? ?ƴ? skin  ???? ?̶??? ????
        # target?? ?????? ?????°??̱⿡, ?????? ?????? target???? ???? ?? ???? ?????? ???? ?ǵ???

        Cycle_loss = (tf.reduce_mean(tf.abs(tf.nn.tanh(fake_A_[:, :, :, 0:3]) - A_batch_images))) \
            * 10.0 + (tf.reduce_mean(tf.abs(tf.nn.tanh(fake_B_[:, :, :, 0:3]) - B_batch_images))) * 10.0
        # Cycle?? ?Ͽ? ???????? ????, Cycle?? ?̹????? high freguency ?????? ???? ???а? ????????????
        ############################ high freqency ???? ?̿? ############################
        
        #f_A_x_, f_A_y = tf.image.image_gradients(tf.nn.tanh(fake_A_[:, :, :, 0:3]))
        #f_A_m_ = tf.add(tf.abs(f_A_x_), tf.abs(f_A_y))
        #f_A_m_low = tf.abs(tf.nn.tanh(fake_A_[:, :, :, 0:3]) - f_A_m_)
        #r_A_m_los = tf.abs(A_batch_images - r_A_m)

        #f_B_x_, f_B_y = tf.image.image_gradients(tf.nn.tanh(fake_B_[:, :, :, 0:3]))
        #f_B_m_ = tf.add(tf.abs(f_B_x_), tf.abs(f_B_y))
        #f_B_m_low = tf.abs(tf.nn.tanh(fake_B_[:, :, :, 0:3]) - f_B_m_)
        #r_B_m_los = tf.abs(B_batch_images - r_B_m)

        #Cycle_loss = (tf.reduce_mean(tf.abs(f_A_m_ - r_A_m))) \
        #    * 5.0 + (tf.reduce_mean(tf.abs(f_B_m_ - r_B_m))) * 5.0 \
        #    + 5.0 + (tf.reduce_mean(tf.abs(f_A_m_low - r_A_m_los))) \
        #    + 5.0 * (tf.reduce_mean(tf.abs(f_B_m_low - r_B_m_los)))
        ################## high freqency ???? ?̿? ##################       # style ?̶??? ????

        G_gan_loss = tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2) \
            + tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2)

        Adver_loss = (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2. \
            + (tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + return_loss + id_loss
        d_loss = Adver_loss

    for (G_grads, G_param) in zip(g_grads,g_trainables_params):

        e_w_g = G_grads * g_grads_scale
        G_param.assign_add(e_w_g)
        e_w_g_buf.append(e_w_g)

    sam_g_grads = g_tape.gradient(g_loss, g_trainables_params)
    for (param, e_w) in zip(g_trainables_params, e_w_g_buf):
        param.assign_sub(e_w)

    for (D_grads, D_param) in zip(d_grads,d_trainables_params):

        e_w_d = D_grads * d_grads_scale
        D_param.assign_add(e_w_d)
        e_w_d_buf.append(e_w_d)

    sam_d_grads = d_tape.gradient(d_loss, d_trainables_params)
    for (param, e_w) in zip(d_trainables_params, e_w_d_buf):
        param.assign_sub(e_w)

    g_optim.apply_gradients(zip(sam_g_grads, g_trainables_params))
    d_optim.apply_gradients(zip(sam_d_grads, d_trainables_params))

    return g_loss, d_loss

def main():

    pre_trained_encoder1 = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    pre_trained_encoder2 = tf.keras.applications.VGG16(include_top=False, input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    pre_trained_encoder2.summary()

    A2B_G_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_G_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_discriminator = F2M_discriminator(input_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))
    A_discriminator = F2M_discriminator(input_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))

    A2B_G_model.summary()
    B_discriminator.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                                   B_discriminator=B_discriminator,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    else:
        A2B_G_model.get_layer("conv_en_1").set_weights(pre_trained_encoder1.get_layer("conv1_conv").get_weights())
        B2A_G_model.get_layer("conv_en_1").set_weights(pre_trained_encoder1.get_layer("conv1_conv").get_weights())
    
        A2B_G_model.get_layer("conv_en_2").set_weights(pre_trained_encoder2.get_layer("block2_conv1").get_weights())
        B2A_G_model.get_layer("conv_en_2").set_weights(pre_trained_encoder2.get_layer("block2_conv1").get_weights())

        A2B_G_model.get_layer("conv_en_3").set_weights(pre_trained_encoder2.get_layer("block3_conv1").get_weights())
        B2A_G_model.get_layer("conv_en_3").set_weights(pre_trained_encoder2.get_layer("block3_conv1").get_weights())


    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):
            min_ = min(len(A_images), len(B_images))
            A = list(zip(A_images, A_labels))
            B = list(zip(B_images, B_labels))
            shuffle(B)
            shuffle(A)
            B_images, B_labels = zip(*B)
            A_images, A_labels = zip(*A)
            A_images = A_images[:min_]
            A_labels = A_labels[:min_]
            B_images = B_images[:min_]
            B_labels = B_labels[:min_]

            A_zip = np.array(list(zip(A_images, A_labels)))
            B_zip = np.array(list(zip(B_images, B_labels)))

            # ?????? ???̿? ???ؼ? distance?? ???ϴ? loss?? ?????ϸ?, ?ᱹ???? ?ش??̹????? ???̸? ?״??? ?????ϴ? ȿ????? ??????????
            gener = tf.data.Dataset.from_tensor_slices((A_zip, B_zip))
            gener = gener.shuffle(len(B_images))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = min_ // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, A_batch_labels, B_batch_images, B_batch_labels = next(train_it)

                g_loss, d_loss = cal_loss(A2B_G_model, B2A_G_model, A_discriminator, B_discriminator,
                                          A_batch_images, B_batch_images, B_batch_labels, A_batch_labels)

                print("Epoch = {}[{}/{}];\nStep(iteration) = {}\nG_Loss = {}, D_loss = {}".format(epoch,step,train_idx,
                                                                                                  count+1,
                                                                                                  g_loss, d_loss))
                
                if count % 100 == 0:
                    fake_B = model_out(A2B_G_model, A_batch_images, False)
                    fake_A = model_out(B2A_G_model, B_batch_images, False)

                    plt.imsave(FLAGS.sample_images + "/fake_B_{}.jpg".format(count), tf.nn.tanh(fake_B[0, :, :, 0:3]) * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/fake_A_{}.jpg".format(count), tf.nn.tanh(fake_A[0, :, :, 0:3]) * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_B_{}.jpg".format(count), B_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_A_{}.jpg".format(count), A_batch_images[0] * 0.5 + 0.5)


                #if count % 1000 == 0:
                #    num_ = int(count // 1000)
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                #    if not os.path.isdir(model_dir):
                #        print("Make {} folder to store the weight!".format(num_))
                #        os.makedirs(model_dir)
                #    ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                #                               A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                #                               g_optim=g_optim, d_optim=d_optim)
                #    ckpt_dir = model_dir + "/F2M_V8_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)

                count += 1

    else:
        if FLAGS.test_dir == "A2B": # train data?? A?? ?ƴ? B?? ?ؾ???
            A_train_data = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
            A_train_data = [FLAGS.A_img_path + data for data in A_train_data]
            A_train_label = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            A_test_data = np.loadtxt(FLAGS.A_test_txt_path, dtype="<U200", skiprows=0, usecols=0)
            A_test_data = [FLAGS.A_test_img_path + data for data in A_test_data]
            A_test_label = np.loadtxt(FLAGS.A_test_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            tr_gener = tf.data.Dataset.from_tensor_slices((A_train_data, A_train_label))
            tr_gener = tr_gener.map(te_input_func)
            tr_gener = tr_gener.batch(1)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((A_test_data, A_test_label))
            te_gener = te_gener.map(te_input_func)
            te_gener = te_gener.batch(1)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_it = iter(tr_gener)
            tr_idx = len(A_train_data) // 1
            te_it = iter(te_gener)
            te_idx = len(A_test_data) // 1

            for i in range(te_idx):
                te_A_images, te_A_labels = next(te_it)
                fake_B, te_feature = model_out(A2B_G_model, te_A_images, False)    # [1, 256]
                te_features = te_feature[0]
                dis = []
                lab = []
                for j in range(tr_idx):
                    tr_A_images, tr_A_labels = next(tr_it)
                    _, tr_feature = model_out(A2B_G_model, tr_A_images, False)    # [1, 256]
                    tr_features = tr_feature[0]

                    d = tf.reduce_sum(tf.abs(tr_features - te_features), -1)
                    dis.append(d.numpy())
                    lab.append(tr_A_labels[0].numpy())

                min_distance = np.argmin(dis, axis=-1)
                generated_age = lab[min_distance]

                name = (A_test_data[i].split("/")[-1]).split(".")[0]
                plt.imsave(FLAGS.fake_B_path + "/" + name + "_{}".format(generated_age) + ".jpg", fake_B[0].numpy() * 0.5 + 0.5)



        if FLAGS.test_dir == "B2A": # train data ?? B?? ?ƴ? A?? ?ؾ???
            B_train_data = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
            B_train_data = [FLAGS.B_img_path + data for data in B_train_data]
            B_train_label = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

            B_test_data = np.loadtxt(FLAGS.B_test_txt_path, dtype="<U200", skiprows=0, usecols=0)
            B_test_data = [FLAGS.B_test_img_path + data for data in B_test_data]
            B_test_label = np.loadtxt(FLAGS.B_test_txt_path, dtype="<U200", skiprows=0, usecols=1)

            tr_gener = tf.data.Dataset.from_tensor_slices((B_train_data, B_train_label))
            tr_gener = tr_gener.map(te_input_func)
            tr_gener = tr_gener.batch(1)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((B_test_data, B_test_label))
            te_gener = te_gener.map(te_input_func)
            te_gener = te_gener.batch(1)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_it = iter(tr_gener)
            tr_idx = len(B_train_data) // 1
            te_it = iter(te_gener)
            te_idx = len(B_test_data) // 1

            for i in range(te_idx):
                te_B_images, te_B_labels = next(te_it)
                fake_A, te_feature = model_out(B2A_G_model, te_B_images, False)    # [1, 256]
                te_features = te_feature[0]
                dis = []
                lab = []
                for j in range(tr_idx):
                    tr_B_images, tr_B_labels = next(tr_it)
                    _, tr_feature = model_out(B2A_G_model, tr_B_images, False)    # [1, 256]
                    tr_features = tr_feature[0]

                    d = tf.reduce_sum(tf.abs(tr_features - te_features), -1)
                    dis.append(d.numpy())
                    lab.append(tr_B_labels[0].numpy())

                min_distance = np.argmin(dis, axis=-1)
                generated_age = lab[min_distance]

                name = (B_test_data[i].split("/")[-1]).split(".")[0]
                plt.imsave(FLAGS.fake_A_path + "/" + name + "_{}".format(generated_age) + ".jpg", fake_A[0].numpy() * 0.5 + 0.5)



if __name__ == "__main__":
    main()