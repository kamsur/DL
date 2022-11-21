import numpy as np
import matplotlib.pyplot as plt

class Checker(object):
    def __init__(self,resolution,tile_size):
        self.resolution=resolution
        self.tile_size=tile_size
        self.output=np.ndarray(0,int)

    def draw(self):
        rep=self.resolution//(2*self.tile_size)
        if self.resolution%(2*self.tile_size)==0:
            self.output=np.concatenate((np.ones(self.tile_size),np.zeros(self.tile_size)))
            tempOutput=np.concatenate((np.zeros(self.tile_size),np.ones(self.tile_size)))
            self.output=np.tile(self.output,(self.tile_size,rep))
            tempOutput=np.tile(tempOutput,(self.tile_size,rep))
            self.output=np.vstack((tempOutput,self.output))
            self.output=np.tile(self.output,(rep,1))
        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(),cmap='gray')
        plt.show()

class Circle(object):
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output=np.ndarray(0,int)

    def draw(self):
        xx = np.linspace(0, self.resolution ,self.resolution) 
        yy = np.linspace(0, self.resolution ,self.resolution)
        x, y = np.meshgrid(xx, yy)
        self.output = (x-self.position[0])**2 + (y-self.position[1])**2
        self.output = self.output < self.radius**2
        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(), cmap = 'gray')
        plt.show()
    
class Spectrum(object):
    def __init__(self, resolution):
        self.resolution = resolution
        self.output=np.ndarray(0,int)
    
    def draw(self):
        a = np.linspace(0, 1, self.resolution)
        b = np.linspace(0, 1, self.resolution)
        a, b = np.meshgrid(a, b)
        self.output = np.dstack((a, b, np.fliplr(a)))
        return self.output.copy()
        
    def show(self):
        plt.imshow(self.draw())
        plt.show()