# Fine-tune Face VGG for Gender Task

There is an extensive writeup of this project at: http://www.eggie5.com/141-Case-Study-Transfer-Learning-for-Gender-Detection

# Usage

## Requirements

* Python 2.7: due to a dependency in the `caffe_tensorflow` project we can't use python 3
* GPU (optional): The model will train and eval much faster if you have a GPU

Here are some python dependencies which can be installed with:

```bash
pip install -r requirements.txt
```



## Files

* train.py - Training CLI

* eval.py - Eval CLI 

* Support Files:

  * metrics.py - Eval routines
  * model.py - Tensorflow Arch
  * data.py - Code for data pipeline
  * Caffe_tensorflow: support for caffe conversion

* Data:

  * VGG_FACE.npy - Caffe Face weights *(for training only)*
  * Training data `fold_[id]_data.txt` (Image paths and Gender labels. Download from: http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/)
  * /data/aligned/ - Directory of gender dataset (Download from: http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.zip)
  * /ckpts/ - weights of Gender model (my model) that I pre-trained *(for eval only)*


User: adiencedb
Password: adience

## Eval

I have included my pre-trained Gender model which can be evaluated using the eval CLI:

```shell
python eval.py --model_path=./ckpts/model.ckpt --val_path=./data/fold_1_data.txt --base_path=./data/aligned/ --batch_size=128
```

## Train

Or you can train (fine-tune) the model from scratch (make sure `base_path` is pointing to the dataset):

```shell
python train.py --model_path=./VGG_FACE.npy --train_path=../data/fold_0_data.txt --val_path=./data/fold_1_data.txt --base_path=./data/aligned/ --batch_size=128 --num_epochs=5
```



