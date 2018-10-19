from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np
import os
import csv
from tqdm import tqdm
import sys

from model import VGG_FACE_16_layers
import metrics
import vgg_preprocessing
import data
import argparse
#get_ipython().run_line_magic('pylab', 'inline')
np.set_printoptions(precision=2)

parser = argparse.ArgumentParser()
parser.add_argument('--val_path', default='data/fold_1_data.txt')
parser.add_argument('--base_path', default="data/aligned/")
parser.add_argument('--prefix', default="landmark_aligned_face.")
parser.add_argument('--model_path', default='./ckpts/model.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)

def main(args):
    val_filenames, val_labels = data.list_images(args.val_path, args.base_path, args.prefix)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(vgg_preprocessing._parse_function, num_parallel_calls=args.num_workers)
    val_dataset = val_dataset.map(vgg_preprocessing.val_preprocess, num_parallel_calls=args.num_workers)
    batched_val_dataset = val_dataset.batch(args.batch_size)

    iterator = tf.data.Iterator.from_structure(batched_val_dataset.output_types,
                                                       batched_val_dataset.output_shapes)
    images, labels = iterator.get_next()

    val_init_op = iterator.make_initializer(batched_val_dataset)


    net = VGG_FACE_16_layers({'input': images})
    logits = net.layers["fc8"]
    
    # Evaluation metrics
    prediction = tf.to_int32(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion_matrix = tf.confusion_matrix(labels, prediction, num_classes=3)
    
    saver = tf.train.Saver()
    
    
    def epoch_eval(sess, matrix_plot=False):
        val_acc, cm = metrics.check_accuracy(sess, correct_prediction, confusion_matrix, val_init_op)
        print('Val accuracy: %f\n' % val_acc)
    
        CLASS_NAMES=["M", "F", "?"]
    
        if matrix_plot:
            plt.figure()
            metrics.plot_confusion_matrix(cm, CLASS_NAMES ,normalize=True,title='Normalized Confusion matrix')
            plt.show()
        else:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)

    with tf.Session() as sess:
        
        #restore model
        print("restoring weights...")
        saver.restore(sess, args.model_path)
        print("...complete")
        
        epoch_eval(sess, matrix_plot=False)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    #python eval.py --model_path=./ckpts/model.ckpt --val_path=../data/fold_1_data.txt --base_path=../data/aligned/