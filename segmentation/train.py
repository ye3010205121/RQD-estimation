# -*- ecoding: utf-8 -*-
# @ModuleName: test02
# @Function: 
# @Author: Jack Chen
# @Time: 2021/8/12 10:13

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
#import keras
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
DATA_DIR = './data/'
img_width = 640
img_height = 480
img_CLASSES = ['background', 'core']    # 图像标注几类，写几个
#train_augmentation = None  # 数据扩增 get_training_augmentation()
train_augmentation = None
val_augmentation = None #get_validation_augmentation()
BACKBONE = 'efficientnetb5'
#BACKBONE = 'seresnext101'
#BACKBONE = 'senet154'
#BACKBONE = 'inceptionresnetv2'
#BACKBONE = 'resnext101'

BATCH_SIZE = 1
CLASSES = ['core']  # 训练时要分出的类
LR = 0.0001 #0.0001
EPOCHS = 80
optim = keras.optimizers.Adam(LR)
#optim = keras.optimizers.Nadam()
# 图像大小在340,341,415,416行设置
checkpoint = "./model_b5_adam.h5"
file_csv = "val_score_b5_adam.csv"
dir = "pr_imgs_2" + "\\"   # 预测图片的保存路径，文件名是从1.png开始往后排



x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(160, 50))
    # print(images.items())
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        # print(image)
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']

    CLASSES = img_CLASSES

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        '''
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch
        '''
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        # newer version of tf/keras want batch to be in tuple rather than list
        return tuple(batch)
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=img_CLASSES)
# dataset = Dataset(x_train_dir, y_train_dir, classes=['core'])

image, mask = dataset[0] # get some sample
visualize(
    image=image,
    background_mask=mask[..., 0].squeeze(),
    core_mask=mask[..., 1].squeeze(),
#     backg4round_mask=mask[..., 2].squeeze(),
)
print(image.shape)
print(mask.shape)

import albumentations as A


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=160, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=160, always_apply=True),

        A.GaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 160)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# Lets look at augmented data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=img_CLASSES, augmentation=train_augmentation)

image, mask = dataset[1] # get some sample
visualize(
    image=image,
    core_mask=mask[..., 0].squeeze(),
    back_mask=mask[..., 1].squeeze(),
#     background_mask=mask[..., 2].squeeze(),
)

import segmentation_models as sm

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`
#
# BACKBONE = 'resnet34'
# BATCH_SIZE = 1
# CLASSES = ['core']
# LR = 0.0001
# EPOCHS = 150

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optomizer
# optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1]))
#focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
#total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

print(img_CLASSES)
# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
#   augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    augmentation=val_augmentation,
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
images_size =  (BATCH_SIZE, img_height, img_width, 3)
masks_size =  (BATCH_SIZE, img_height, img_width, n_classes)
assert train_dataloader[0][0].shape == images_size
assert train_dataloader[0][1].shape == masks_size

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint, save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
    verbose=1,
)

import pandas as pd
tr_epoch = history.epoch
val_loss = history.history['val_loss']
#val_precision = history.history['val_precision']
val_f1score = history.history['val_f1-score']
val_iou = history.history['val_iou_score']

ch = []
for i in tr_epoch:
    i = int(i)
    epoch = tr_epoch[i]
    val_loss_1 =val_loss[i]
#     val_precision_1 = val_precision[i]
    val_f1score_1 = val_f1score[i]
    val_iou_1 = val_iou[i]
    ch.append([epoch, val_loss_1,  val_f1score_1, val_iou_1])

column = ["Epoch", "val_loss",  "val_f1score", "val_IoU"]
rect = pd.DataFrame(ch, columns=column)
# file_csv = "val_score_res34_adam.csv"
rect.to_csv(file_csv, index=False, header=True)
print("保存文件成功")

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=val_augmentation,
#     augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert test_dataloader[0][0].shape == images_size
assert test_dataloader[0][1].shape == masks_size

# load best weights
model.load_weights(checkpoint)


scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

n = 1
# ids = np.random.choice(np.arange(len(test_dataset)), size=n)
print(len(test_dataset),0)

for i in range(len(test_dataset)):
    image, mask = test_dataset[i] # get some sample:
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)
    image = pr_mask.squeeze()
    # dir = "pr_imgs" + "\\"
    filename = dir + str(i) + ".png"
    print(filename)
    cv2.imwrite(filename, image)
