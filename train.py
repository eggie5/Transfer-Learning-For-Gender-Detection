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
parser.add_argument('--train_path', default='data/fold_0_data.txt')
parser.add_argument('--val_path', default='data/fold_1_data.txt')
parser.add_argument('--base_path', default="data/aligned/")
parser.add_argument('--prefix', default="landmark_aligned_face.")
parser.add_argument('--model_path', default='./VGG_FACE.npy', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=10, type=int)




    
def main(args):
    
    print("loading dataset...")

    train_filenames, train_labels = data.list_images(args.train_path, args.base_path, args.prefix)
    val_filenames, val_labels = data.list_images(args.val_path, args.base_path, args.prefix)
    data_len=len(val_filenames)
    
    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=512) 
    train_dataset = train_dataset.map(vgg_preprocessing._parse_function, num_parallel_calls=args.num_workers)
    train_dataset = train_dataset.map(vgg_preprocessing.training_preprocess, num_parallel_calls=args.num_workers)
    batched_train_dataset = train_dataset.batch(args.batch_size)
    batched_train_dataset = batched_train_dataset.prefetch(2)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(vgg_preprocessing._parse_function, num_parallel_calls=args.num_workers)
    val_dataset = val_dataset.map(vgg_preprocessing.val_preprocess, num_parallel_calls=args.num_workers)
    batched_val_dataset = val_dataset.batch(args.batch_size)
    batched_val_dataset = batched_val_dataset.prefetch(2)

    iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                       batched_train_dataset.output_shapes)
    images, labels = iterator.get_next()

    train_init_op = iterator.make_initializer(batched_train_dataset)
    val_init_op = iterator.make_initializer(batched_val_dataset)
    
    print("...done")

    print("Building graph...")
    net = VGG_FACE_16_layers({'input': images})
    logits = net.layers["fc8"]

    fc8_variables = tf.contrib.framework.get_variables('fc8')
    fc8_init = tf.variables_initializer(fc8_variables)


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    opt = tf.train.AdamOptimizer()
    fc8_train_op = opt.minimize(loss, var_list=fc8_variables) #we only want to update FC8 ala fine-tuning
    
    # Evaluation metrics
    prediction = tf.to_int32(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    confusion_matrix = tf.confusion_matrix(labels, prediction, num_classes=3)
    
    saver = tf.train.Saver()
    print("...done")
    
    
    def epoch_eval(sess, matrix_plot=False):
        steps = (data_len//args.batch_size)+1
        # Check initial accuracy
        train_acc, train_cnf_matrix = metrics.check_accuracy(sess, correct_prediction, confusion_matrix, train_init_op, total=steps)
        val_acc, cm = metrics.check_accuracy(sess, correct_prediction, confusion_matrix, val_init_op, total=steps)
        print('\nTrain accuracy: %f' % train_acc)
        print('\nVal accuracy: %f\n' % val_acc)
    
        CLASS_NAMES=["M", "F", "?"]
    
        if matrix_plot:
            plt.figure()
            metrics.plot_confusion_matrix(val_cnf_matrix, CLASS_NAMES ,normalize=True,title='Normalized Confusion matrix')
            plt.show()
        else:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)

    with tf.Session() as sess:

        # Load the data
        print("Loading caffe weights...")
        sess.run(tf.global_variables_initializer())
        sess.run(fc8_init)  # initialize the new fc8 layer
        net.load(args.model_path, sess, scratch_layers=["prob", "fc8"]) #restore weights except for last FC
        print("...done")
    

        print("initial eval (random init)...")
        epoch_eval(sess)

        for epoch in range(args.num_epochs):
            print('\nStarting epoch %d / %d' % (epoch + 1, args.num_epochs))
            sess.run(train_init_op)
    
            epoch_losses=[]
    
            pbar = tqdm(total = (data_len//args.batch_size)+1)
            while True:
                try:
                    _, xent = sess.run([fc8_train_op, loss])
                    epoch_losses.append(xent)
                    pbar.update(1)
                    pbar.set_description("mean epoch loss(xent): %f" % (np.mean(epoch_losses)))
                except tf.errors.OutOfRangeError:
                    break
            
            pbar.close()
            #batch eval
            epoch_eval(sess)

        #save model
        print("Saved model to ckpts dir")
        save_path = saver.save(sess, "./ckpts/model.ckpt")
    
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    # python train.py --model_path=./VGG_FACE.npy --train_path=../data/fold_0_data.txt --val_path=../data/fold_1_data.txt --base_path=../data/aligned/ --batch_size=64
    
    # python train.py --model_path=../transferlearning/caffe-tensorflow/VGG_FACE.npy --train_path=../transferlearning/data/fold_0_data.txt --val_path=../transferlearning/data/fold_1_data.txt --base_path=../transferlearning/data/aligned/ --batch_size=128 --num_epochs=5
