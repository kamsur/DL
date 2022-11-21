import os
import json
import scipy.misc
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        if os.path.exists(str(file_path)):
            self.file_path=str(file_path)
            self.file_names=np.array(os.listdir(self.file_path))
        else:
            self.file_path=""
            self.file_names=np.array([])
        if os.path.exists(str(label_path)):
            f=open(str(label_path))
            self.label_dict=(json.load(f))
            f.close()
        else:
            self.label_dict=dict()
        self.batch_size=int(batch_size)
        self.image_size=list(image_size)
        self.rotation=rotation
        self.mirroring=mirroring
        self.shuffle=shuffle
        if self.shuffle:
            np.random.shuffle(self.file_names)
        self.epoch_number=-1
        self.file_pos=0

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        images=list()
        labels=list()
        if self.file_pos==0:
            self.epoch_number+=1
            if self.shuffle:
                np.random.shuffle(self.file_names)
        flag=0
        for i in range(self.batch_size):
            if self.file_pos==len(self.file_names):
                self.file_pos=0
                flag+=1 
            image=np.load(os.path.join(self.file_path,self.file_names[self.file_pos]),mmap_mode='r')
            image=skimage.transform.resize(image,self.image_size)
            image=self.augment(image)
            label=self.label_dict[os.path.splitext(self.file_names[self.file_pos])[0]]
            images.append(image)
            labels.append(label)
            self.file_pos+=1         
        if flag!=0 or self.file_pos==len(self.file_names):
            self.file_pos=0
        return np.array(images), np.array(labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring and np.random.choice(np.array([0,1])):
            img=np.fliplr(img)
        if self.rotation:
            img=np.rot90(img,np.random.choice(np.array([0,1,2,3])))

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_number

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        batch=self.next()
        for i in range(self.batch_size):
            plt.subplot(self.batch_size//5+1,5,i+1)
            plt.imshow(batch[0][i])
            plt.title(self.class_name(batch[1][i]))
            plt.axis('off')
        plt.show()


