import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import PySimpleGUI as sg
import sys
import heapq

#%% load models
"""
All the models should be in ./models/folder
"""
print("Loading Models")

inception_model = tf.keras.models.load_model('models/InceptionV3_Serengeti')
print(" Loaded InceptionV3 ")
xception_model = tf.keras.models.load_model('models/Xception_Serengeti') #Re-run train_model.py by changing pre-trained model to Xception
print(" Loaded Xception ")
resnet_model = tf.keras.models.load_model('models/ResNet50_Serengeti') #Re-run train_model.py by changing pre-trained model to Xception
print(" Loaded ResNet50 ")
print("Select an image and model to classify")

#%%
flag = True
while(flag):
    if len(sys.argv) == 1:
        event, values = sg.Window('Serengeti_GUI',
                        [[sg.Text('Select an Image of animal from trained classes')],
                        [sg.In(), sg.FileBrowse()],
                        [sg.Text('Select a model')],
                        [sg.InputCombo(('', 'InceptionV3', 'Xception','ResNet50'), size=(20, 4))],
                        [sg.Open(), sg.Cancel()]]).read(close=True)
        img_path = values[0]
        model_selected = values[1]
    else:
        flag = False
    
    print(f'img_path: {img_path}')
    print(f'Model Selected: {model_selected}')
# %%
"""
Please add class names that your trained data contains here
"""
    labels = np.array(['baboon', 'buffalo', 'cheetah', 'eland', 'giraffe', 'guineafowl',
           'hippopotamus', 'hyenaspotted', 'impala', 'lionfemale', 'reedbuck',
           'secretarybird', 'serval', 'topi', 'warthog']) #Serengeti sample class names

# %%
    if(model_selected in ['ResNet50','InceptionV3', 'Xception']):
        try:
            img = Image.open(img_path)
            img = img.resize((224,224))
            img = np.asarray(img)*(1.0/255)
            img = img[:,:,:3]
        except Exception:
            flag = False
            
        
        if(model_selected == "ResNet50"):
            predicted_label = resnet_model.predict(np.array([img]))
        
        elif(model_selected == "InceptionV3"):
            predicted_label = inception_model.predict(np.array([img]))
        
        elif(model_selected == "Xception"):
            predicted_label = xception_model.predict(np.array([img]))
        
        print(predicted_label)
        # %%
        idx = np.argmax(predicted_label, axis=1)
        print("Predicted as: ",labels[idx])
        top5_labels = labels[heapq.nlargest(5, np.arange(len(labels)), predicted_label[0].take)]
        top5_prob = heapq.nlargest(5, predicted_label[0])
        
        print(f'top5 labels: {top5_labels}')
        print(f'top5 probabilities: {top5_prob}')
        
        # %%
        plt.figure(figsize=(10,10))
        plt.subplot(2,1,1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title("Detected as "+str(labels[idx][0])+" by "+model_selected)
        
        plt.subplot(2,1,2)
        plt.bar(np.arange(5),top5_prob, tick_label=top5_labels)
        plt.title("Top 5 predictions and probabilities")
        plt.show()
    else:
        flag = False


