# CHALLENGE
Here is the challenge.  Also, you should be able to figure out a way to do this challenge without using a lot of computational resources.

1. Model Conversion: Convert the VGG face descriptor model http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  to Tensorflow format.
2. Transfer Learning: Using the above Convolutional Neural Network as a feature descriptor, build a classifier for the gender dataset (http://www.openu.ac.il/home/hassner/Adience/data.html#agegender). You can alternatively access the same dataset through the link  http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/

You are free to use open source tools and code snippets but please do cite any external resources used.

Submission Instructions
Submit:
Final Tensorflow classifier model trained on the gender dataset (architecture and weights)
Code for training and evaluating the model
Results/metrics (for all the classes and overall) obtained for the trained model
Readme file briefly listing the steps taken and the steps to run your code (training and evaluation)

Combine all the files into a tar bundle and submit the tar file to dropbox by going to the link -
https://www.dropbox.com/request/itRKx41NA9eMQo6B340K


User: adiencedb
Password: adience

```
docker run -ti -v ~/Development/workspace/matroid/vgg_face_caffe/:/root/shared_folder bvlc/caffe:cpu bash
```

Upgrade the old Caffe Syntax: https://github.com/ethereon/caffe-tensorflow/issues/39
```
upgrade_net_proto_text shared_folder/VGG_FACE_deploy.prototxt shared_folder/VGG_FACE_deploy.prototxt2
```

Run conversion:

```
caffe-tensorflow/convert.py shared_folder/VGG_FACE_deploy.prototxt2 --code-output-path vVGG_FACE_deploy.py --caffemodel shared_folder/VGG_FACE.caffemodel --data-output-path VGG_FACE.npy
```

```
Type                 Name                                          Param               Output
----------------------------------------------------------------------------------------------
Input                input                                            --     (1, 3, 224, 224)
Convolution          conv1_1                                          --    (1, 64, 224, 224)
Convolution          conv1_2                                          --    (1, 64, 224, 224)
Pooling              pool1                                            --    (1, 64, 112, 112)
Convolution          conv2_1                                          --   (1, 128, 112, 112)
Convolution          conv2_2                                          --   (1, 128, 112, 112)
Pooling              pool2                                            --     (1, 128, 56, 56)
Convolution          conv3_1                                          --     (1, 256, 56, 56)
Convolution          conv3_2                                          --     (1, 256, 56, 56)
Convolution          conv3_3                                          --     (1, 256, 56, 56)
Pooling              pool3                                            --     (1, 256, 28, 28)
Convolution          conv4_1                                          --     (1, 512, 28, 28)
Convolution          conv4_2                                          --     (1, 512, 28, 28)
Convolution          conv4_3                                          --     (1, 512, 28, 28)
Pooling              pool4                                            --     (1, 512, 14, 14)
Convolution          conv5_1                                          --     (1, 512, 14, 14)
Convolution          conv5_2                                          --     (1, 512, 14, 14)
Convolution          conv5_3                                          --     (1, 512, 14, 14)
Pooling              pool5                                            --       (1, 512, 7, 7)
InnerProduct         fc6                                              --      (1, 4096, 1, 1)
InnerProduct         fc7                                              --      (1, 4096, 1, 1)
InnerProduct         fc8                                              --      (1, 2622, 1, 1)
Softmax              prob                                             --      (1, 2622, 1, 1)
Converting data...
Saving source...
Done.
```
## References

Project using the dataset for age prediction TF model:
https://github.com/lizihaoleo/CNN-model-for-age-prediction/blob/master/AgeClassifier.ipynb

FTP Access to the data
https://blog.csdn.net/Allyli0022/article/details/53696068

VGG-Face Keras
https://github.com/rcmalli/keras-vggface

Convert Caffe to TF:
https://ndres.me/post/convert-caffe-to-tensorflow/

Seems like cononical repo for caffe-> conversion:
https://github.com/ethereon/caffe-tensorflow

read matlab weights into TF:
https://github.com/ZZUTK/Tensorflow-VGG-face/blob/master/test_vgg_face.py