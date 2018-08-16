"""
Do segmentation
"""
from __future__ import print_function

import argparse
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc
import cv2

from model import ICNet
import time


class Segment(object):

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        self.model = None

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._load_model()

    def _load_model(self):

        if self.model_name == 'icnet':
            self.model = ICNet()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.model.load(self.model_path, self.sess)
            print('Model initialed.')
        else:
            print('Only ICNet supported.')

    def seg_img_f(self, img_f, save_dir, is_show=False):
        tic = time.time()
        self.model.read_input(img_f)
        preds = self.model.forward(self.sess)

        original_img = cv2.imread(img_f)
        original_img = np.array(original_img, dtype='float32')
        mask_color = np.array(preds[0], dtype='float32')

        overlayed_img = cv2.addWeighted(original_img, 0.4, mask_color, 0.6, 0)

        target_f = os.path.join(save_dir,
                                '{}_seg.png'.format(os.path.basename(img_f).split('.')[0]))
        cv2.imwrite(target_f, overlayed_img)

        # print('saved into {}'.format(target_f))
        print('fps: ', round(1/(time.time() - tic), 3))
        if is_show:
            cv2.imshow('o', overlayed_img)
            cv2.waitKey(0)

    def seg_img(self, img):
        # img should read with opencv in RGB format
        self.model.read_image(img)
        preds = self.model.forward(self.sess)
        original_img = np.array(img, dtype='float32')
        mask_color = np.array(preds[0], dtype='float32')

        overlayed_img = cv2.addWeighted(original_img, 0.4, mask_color, 0.6, 0)
        return overlayed_img, preds

    def seg_video(self, video_f, is_save=True, is_record=False):
        if os.path.exists(video_f):
            save_dir = os.path.join(os.path.dirname(video_f), os.path.basename(video_f).split('.')[0])
            save_record_f = os.path.join(os.path.dirname(video_f), os.path.basename(video_f).split('.')[0] + '.mp4')
            i = 0
            cap = cv2.VideoCapture(video_f)

            while cap.isOpened():
                ret, frame = cap.read()
                tic = time.time()
                if ret:
                    i += 1
                    res, _ = self.seg_img(frame)

                    if is_save:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        cv2.imwrite(os.path.join(save_dir, 'frame_%04d.jpg' % i), res)
                    if is_record:
                        # TODO: do some record things
                        pass
                    print('fps: ', round(1 / (time.time() - tic), 4))
                    cv2.imshow('seg', res)
                    cv2.waitKey(1)
        else:
            print('# video file not exist: '.format(video_f))

    def seg_dir(self, img_dir):
        all_images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith('jpg') or i.endswith('png')
                      or i.endswith('jpeg')]

        save_dir = 'results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img in all_images:

            self.seg_img_f(img, save_dir)


if __name__ == '__main__':
    seg = Segment('icnet', 'model/cityscapes/icnet.npy')
    # seg.seg_img_f('input/cityscape.png')
    # seg.seg_dir('/media/jintian/sg/permanent/Cityscape/leftImg8bit/demoVideo/stuttgart_00')
    seg.seg_video('/media/jintian/sg/permanent/Cityscape/test.mp4')