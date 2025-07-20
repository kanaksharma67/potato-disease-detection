import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


EPOCHS=50
data=tf.data.Datasets.image_datset_from_directory(
    'PlantVillage',
    shuffle=True,
    image_size=(256,256),
    batch_size=32
)

print(data)
classNames=data.class_names
print(classNames)


for img_batch,label_batch in data.take(1):
    print(img_batch.shape)
    print(label_batch.shape.numpy())
    # ax=plt.subplots(3,4,i+1)
    plt.imshow(img_batch[0].numpy().astype("uint8"))
    plt.title(classNames[label_batch[0]])


#80%===>training
# 20%===>10%vlaidation, 10%test
# len(data)*train_ds=0.8
# train_size=int(len(data)*0.8)
# train_ds=data.take(54)
# len(train_ds)
# test_ds=data.skip(54)
# len(test_ds)*0.1=====>6.8888
# val_ds=test_ds.take(6)
# actual_test_ds=test_ds.skip(6)



def train_test(ds,train_split=0.8,val_splot=0.1,test_split=0.1,shuffle_size=10000):
    ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(len(ds)*train_split)
    train_ds=ds.take(train_size)
    test_ds=ds.skip(train_size)
    val_size=int(len(test_ds)*val_splot)
    val_ds=test_ds.take(val_size)
    actual_test_ds=test_ds.skip(test_split)
    return train_ds,val_ds,actual_test_ds

train_ds,val_ds,test_ds=train_test(data)

train_ds=train_ds.cache('train_ds').shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache('val_ds').shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache('test_ds').shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


resize_and_rescale=tf.keras.Sequential([
    layers.Resizing(256,256),
    layers.Rescaling(1.0/255.0)
])

data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),  
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomContrast(0.2)
])

model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),padding='same',activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),
    #dense layers
    layers.Flatten(),
    layers.Dense(32,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(classNames),activation='softmax')
    #softmax will normalize the probabiity of your classes


])


model.build((32,256,256,3))
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(train_ds,epochs=EPOCHS,validation_data=val_ds,batch_size=32,verbose=1)
