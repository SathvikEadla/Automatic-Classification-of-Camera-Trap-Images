import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.externals.joblib import load, dump
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

path =  '/raid/Serengeti/Mini_Project/Data' #path to dataset
#%%
def load_train_images(path_dir):
    file_path = []
    labels = []
    print(('Going to read images from ', path_dir))
    class_dirs = os.listdir(path_dir) 
    for val in class_dirs:
        current_dir = os.path.join(path_dir, val,'*.jpg')
        ls = glob.glob(current_dir)        
        for fl in ls:
            labels.append(val)           
        file_path += ls
    print('Done reading.')
    return labels, file_path
    
labels, images = load_train_images(path)

nClasses = len(set(labels))
la = np.arange(0,nClasses)
lb = LabelBinarizer()
lb.fit(la)
le = LabelEncoder()
le.fit(labels)
le_transformed = le.transform(labels)

images = np.array(images)*(1.0/255)
size = len(images)
idx = np.arange(0,size)

train_size = int(len(idx)*0.9)
train_idx = np.random.choice(idx, train_size)
test_idx = np.setdiff1d(idx, train_idx)

xtrain_images = images[train_idx]
ytrain_labels = lb.transform(le_transformed[train_idx])

xtest_images = images[test_idx]
ytest_labels = lb.transform(le_transformed[test_idx])

#%%

inceptionv3_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3)) #change here to use any pre-trained model
inceptionv3_model.trainable = False

#%%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(1024,activation='relu')
prediction_layer = tf.keras.layers.Dense(nClasses, activation='softmax')

inputs = tf.keras.Input(shape=(224,224,3))
x = inceptionv3_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = dense_layer(x)
outputs = prediction_layer(x)

model_inception = tf.keras.Model(inputs, outputs)

#%%
base_learning_rate = 0.001
model_inception.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model_inception.summary()

#%%
epochs = 50
batch_size = 32
history_inception = model_inception.fit(x = xtrain_images,
                    y = ytrain_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split = 0.1
                    )

#%%
model_inception.save('InceptionV3_Serengeti')

#%%

#####################################################################################################################
#                Metrics
#####################################################################################################################
acc = history_inception.history['accuracy']
val_acc = history_inception.history['val_accuracy']

loss=history_inception.history['loss']
val_loss=history_inception.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.ylim([0,1.0])

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.ylim([0,1.0])

plt.savefig('InceptionV3_Serengeti_acc_loss.png',dpi =300)
#%%
#####################################################################################################################
#                Confusion Matrix
#####################################################################################################################

real_labels = lb.inverse_transform(ytrain_labels)
predicted_labels = np.argmax(model_inception.predict(xtrain_images), axis=1)

#%%
cm = np.asarray(tf.math.confusion_matrix(real_labels, predicted_labels))
cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
#%%
fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(cm)


ax.set_xticks(np.arange(nClasses))
ax.set_yticks(np.arange(nClasses))

ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)


plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(nClasses):
    for j in range(nClasses):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion Matrix for InceptionV3 model")
fig.tight_layout()
plt.savefig("InceptionV3_Serengeti_ConfusionMatrix.png", dpi = 300)