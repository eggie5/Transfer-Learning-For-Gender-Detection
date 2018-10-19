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
pip install -r requirements.txt
```

# Report

## Caffe to TF Conversion

I've never converse a Caffe project before, but I know a popular solution in the community is the `caffe-tensorflow` project. In order to do the conversion you need to have a working Caffe environment w/ python bindings. I used the docker image `blvc/caffe` :

```shell
docker run -ti -v ~/Development/workspace/matroid/vgg_face_caffe/:/root/shared_folder bvlc/caffe:cpu bash
```

First, we need to update the legacy Caffe syntax that comes with the VGG Face distribution:
```shell
upgrade_net_proto_text VGG_FACE_deploy.prototxt VGG_FACE_deploy.prototxt2
```

Run conversion:

```shell
caffe-tensorflow/convert.py VGG_FACE_deploy.prototxt2 --code-output-path VGG_FACE_deploy.py --caffemodel VGG_FACE.caffemodel --data-output-path VGG_FACE.npy
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


The output of this conversion is an graph file and weights:`VGG_FACE_deploy.py` and `VGG_weights.npy`

```python
from kaffe.tensorflow import Network
class VGG_FACE_16_layers(Network):
    def setup(self):
        (self.feed('input')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(2622, relu=False, name='fc8')
             .softmax(name='prob'))
```



This is the VGG architecture w/ a 2622D softmax layer for the original face detection task. 

We can load the pre-trained model like this:

```python
net = VGG_FACE_16_layers({'input': images})
```

Where `input` is the graphs input placeholder.



## Transfer Learning

The task now is to employ transfer learning  on the pertained model from the face detection domain to the gender domain. We can do that by fine-tuning the model on the Gender Dataset.

### Fine Tuning

First we need to modify the final softmax layer of the network to have only 3 classes: Male, Female or Unknown. Then we need to retrain the last few layers of the network w/ the new dataset. This is where there are a number of techniques for retraining. 

* Retrain only the softmax layer
* Retrain any number of the FC layers
* Retrain all the layers (w/ a slow learning rate)

Or you can do a combination of these techniques, for example, first retrain the FCs for a few epochs, then retrain the whole network for a few epochs. 

Regardless here is the general technique:

1. Add your custom network on top of an already-trained base network. 
2. Freeze the base network. 
3. Train the part you added. 
4. Unfreeze some layers in the base network. 
5. Jointly train both these layers and the part you added. 

## Evaluation

We can look at standard multi-class evaluation techniques like top-k accuracy. However, in this case there are only 3 classes so we'll just look at overall accuracy. However, depending our our requirements, we might be interested in other characteristics such as false-postive rates. For example, is it more expensive to misclassify a woman as a man or a man as a women? We can get insights into this by looking at the confusion matrix:

![FaceNet](https://eggie5_production.s3.amazonaws.com/static/213853.png)

For example, on the 5th epoch, you can see the overall accuracy was 84% but is that good or not? We can see that 90% of the time we can correctly classify a Male and that 85% of the time we can correctly classify a Female. However, 16% of the time we confuse a Female as a Male, but not as frequently the other way around which I think is intuitive b/c men have long hair much more frequently than women having short. Maybe your business objective is dependent on not minimizing Female confusion. 

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