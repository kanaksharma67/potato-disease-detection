import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Constants ---
EPOCHS = 50
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123
data_dir = "PlantVillage"

# --- Train/Validation Split ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# --- Optimize Dataset ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# --- Model ---
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    data_augmentation,
    layers.Rescaling(1.0 / 255.0),  # normalize [0,255] to [0,1]

    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

print(model.summary())

# --- Compile ---
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

scores=model.evaluate(test_ds)

his=history.history.keys()
acc=history.history['accuracy']
val_acc=history.history['val_accuracy'] 
print("Test Loss:", scores[0])


for image_batch , label_batch in test_ds.take(1):
    predictions = model.predict(image_batch[0])
    predicted_labels = tf.argmax(predictions, axis=1)
    print("Predicted labels:", predicted_labels.numpy())
    print("True labels:", label_batch[0].numpy())
model_version=1
model.save(f"../models/plant_disease_model_v{model_version}.h5")
