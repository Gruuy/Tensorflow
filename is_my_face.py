#!usr/bin/python3
#coding=utf-8

import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
 
 
size=64
 
input = tf.placeholder(tf.float32,[None,size,size,3])
output = tf.placeholder(tf.float32,[None,2])#输出加两个，true or false
#这里注意的是tf.reshape不是np.reshape
# images = tf.reshape(input,[-1,size,size,3])
 
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)
 
#下面开始进行卷积层的处理
#第一层卷积，首先输入的图片大小是64*64
def cnnlayer():
    #第一层卷积
    conv1 = tf.layers.conv2d(inputs=input,
                            filters=32,
                            kernel_size=[5,5],
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu)#(64*64*32)
#第一层池化
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2)#(32*32*32)
 
#第二层卷积
    conv2 = tf.layers.conv2d(inputs=pool1,
                            filters=32,
                            kernel_size=[5,5],
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu)#(32*32*32)
 
#第二层池化
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2)#(16*16*32)
 
#第三层卷积
    conv3 = tf.layers.conv2d(inputs=pool2,
                            filters=32,
                            kernel_size=[5,5],
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu)#(变成16*16*32)
#第三层池化
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2,2],
                                    strides=2)#(8*8*32)
 
#第四层卷积
    conv4 = tf.layers.conv2d(inputs=pool3,
                             filters=64,
                             kernel_size=[5,5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)#(变成8*8*64）

 
#卷积网络在计算每一层的网络个数的时候要细心一些
#卷积层加的padding为same是不会改变卷积层的大小的
#要注意下一层的输入是上一层的输出
#平坦化
    flat = tf.reshape(conv4,[-1,8*8*64])
 
#经过全连接层
    dense = tf.layers.dense(inputs=flat,
                            units=4096,
                            activation=tf.nn.relu)
 
#drop_out，flat打错一次
    drop_out = tf.layers.dropout(inputs=dense,rate=0.2)
 
#输出层
    logits = tf.layers.dense(drop_out,units=2)
    return logits
    # yield logits
out = cnnlayer()
# out = next(cnnlayer())
predict = tf.argmax(out,1)
saver = tf.train.Saver()
sess = tf.Session()
#加载模型
saver.restore(sess,tf.train.latest_checkpoint('./'))
 
def is_my_face(image):
    res = sess.run(predict, feed_dict={input: [image / 255.0]})
    if res[0] == 1:
        return True
    else:
        return False
 
classfier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
cam = cv2.VideoCapture(0)
 
while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=classfier.detectMultiScale(gray_image,1.2,3,minSize=(64,64))
    if not len(faces):
        # print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            sys.exit(0)
 
    for x,y,w,h in faces:
        face=img[y:y+h,x:x+w]
        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))
        if(is_my_face(face)):
            cv2.rectangle(img, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            cv2.putText(img,"Gruuy",(x+30,y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),4)
            cv2.imshow('img', img)
        else:
            cv2.imshow('img', img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            sys.exit(0)
 
sess.close()
