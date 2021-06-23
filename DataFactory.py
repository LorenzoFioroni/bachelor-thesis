from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch


# Super class with the basic definition of a dataSet. 
# It generates the points when the class is instantiated. For problems 
# of classification in 1 to 4 classes it also generates the states 
# associated with each class and the overlaps between them. 
class DataSet:

    def __init__(self, name, dim, nClasses, nTrain, nValid, nTest, seed = None):
        self.name = name         # name of the dataSet
        self.nTrain = nTrain     # number of elements in training set
        self.nValid = nValid     # number of elements in validation set
        self.nTest = nTest       # number of elements in test set
        self.dim = dim           # dimensionality
        self.nClasses = nClasses # number of classes

        if seed: torch.manual_seed(seed)

        # Table of (nTrain+nValid+nTest)x(dim) points randomly generated
        # with a uniform distribution between -1 and 1
        size = (self.nTrain+self.nValid+self.nTest, self.dim)
        self.ds = 2 * torch.rand(size=size, requires_grad=False) - 1 
    

    # Function to get the labelStates representing the classes and their overlaps
    # with each other. For 2, 3 or 4 classes generates the states in the form 
    # |Ψ_class> = a|0> + b|1> and returns the Hermitian matrix |Ψ_class><Ψ_class| 
    # associated. The expected fidelity is defined as |<Ψ_class_1|Ψ_class_2>|^2.
    def getStates(self):
       
        if self.nClasses == 2:
            states = [
                np.array([1, 0]).reshape(2,1),
                np.array([0, 1]).reshape(2,1)
            ]
        elif self.nClasses == 3:
            states = [
                np.array([1, 0]).reshape(2,1),
                np.array([1/2, np.sqrt(3)/2]).reshape(2,1),
                np.array([1/2, -np.sqrt(3)/2]).reshape(2,1)
            ]
        elif self.nClasses == 4:
            theta = np.arccos(-1/3)
            phi = np.arccos(-np.sqrt(2/9)/np.sin(theta))
            states = [
                np.array([1, 0]).reshape(2,1),
                np.array([np.cos(theta/2), np.sin(theta/2)]).reshape(2,1),
                np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)]).reshape(2,1),
                np.array([np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]).reshape(2,1)
            ]
        else:
            raise Exception("Cannot generate label states for {} classes yet".format(self.nClasses))

        labelStates = torch.tensor(np.array([s*np.conj(s).T for s in states]), requires_grad = False)
        
        expectedFidelity = torch.empty(self.nClasses, self.nClasses)
        for i in range(self.nClasses):
            for j in range(i, self.nClasses):
                expectedFidelity[i, j] = expectedFidelity[j, i] = np.vdot(states[i], states[j])*np.vdot(states[j], states[i])
        
        return labelStates, expectedFidelity
         
    
    # Returns the first self.nTrain points
    def getTrain(self):
        return self.ds[:self.nTrain,:],self.label[:self.nTrain]
    

    # Returns self.nValid points after the first self.nTrain
    def getValid(self):
        return self.ds[self.nTrain:self.nTrain+self.nValid,:], self.label[self.nTrain:self.nTrain+self.nValid]
    

    # Returns the last self.nTest points
    def getTest(self):
        return self.ds[self.nTrain+self.nValid:,:], self.label[self.nTrain+self.nValid:]


    # Simple utility function to visualize the whole set of points for a 2D problem
    # (correctly categorized)   
    def plot(self):
        if self.dim != 2: raise Exception("Not a 2D dataSet")
        plt.scatter(self.ds[:, 0], self.ds[:, 1], c=self.label, s = 6, cmap="Accent") 
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(self.name)

# --------------------

# 2D dataSet. Classification in 2 classes.
# The points inside the circle of radius (2/pi)^(1/2) are associated with the 
# first class, the others with the second one.
class Circle(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        self.radius = np.sqrt(2/np.pi)
        DataSet.__init__(self, "circle", 2, 2, nTrain, nValid, nTest, seed)
        self.label = (self.ds[:, 0]**2 + self.ds[:, 1]**2 < self.radius**2).type(torch.int64) 

    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
        theta = np.linspace(0, 2*np.pi, 100)
        return [self.radius*np.cos(theta)], [self.radius*np.sin(theta)]


# 2D dataSet. Classification in 4 classes.
# There are three non overlapping circles. The points inside those circles are 
# associated with one of the first three classes. The points outside every circle
# (background) are associated to the fourth class.
class ThreeCircles(DataSet): 

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        DataSet.__init__(self, "three circles", 2, 4, nTrain, nValid, nTest, seed)
        self.cent1 = (-1, 1)
        self.r1 = 1
        self.cent2 = (-0.5, -0.5)
        self.r2 = 0.5
        self.cent3 = (1, 0)
        self.r3 = 1
        c1 = ((self.ds[:, 0]-self.cent1[0])**2 + (self.ds[:, 1]-self.cent1[1])**2 < self.r1**2).type(torch.int64) 
        c2 = ((self.ds[:, 0]-self.cent2[0])**2 + (self.ds[:, 1]-self.cent2[1])**2 < self.r2**2).type(torch.int64) 
        c3 = ((self.ds[:, 0]-self.cent3[0])**2 + (self.ds[:, 1]-self.cent3[1])**2 < self.r3**2).type(torch.int64) 
        self.label = c1+2*c2+3*c3
    
    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
        theta = np.linspace(0, 2*np.pi, 100)
        return [self.r1*np.cos(theta)+self.cent1[0], self.r2*np.cos(theta)+self.cent2[0], self.r3*np.cos(theta)+self.cent3[0]],[self.r1*np.sin(theta)+self.cent1[1], self.r2*np.sin(theta)+self.cent2[1], self.r3*np.sin(theta)+self.cent3[1]]


# 4D dataSet. Classification in 2 classes.
# The points inside the hypersphere of radius (2/pi)^(1/2) are associated with the 
# first class, the others with the second one.
class FourDimHypersphere(DataSet):
    
    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        self.radius = np.sqrt(2/np.pi)
        DataSet.__init__(self, "four dimensional hypersphere", 4, 2, nTrain, nValid, nTest, seed)
        self.label = (self.ds[:, 0]**2 + self.ds[:, 1]**2 + self.ds[:, 2]**2 + self.ds[:, 3]**2 < self.radius**2).type(torch.int64) 


# 2D dataSet. Classification in 3 classes.
# The points inside the circle of radius (0.8-2/pi)^(1/2) are associated with the 
# first class, the points outside the circle of radius (2/pi)^(1/2) with the second
# one and the remaining points are assigned to the third class.
class Annulus(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        DataSet.__init__(self, "annulus", 2, 3, nTrain, nValid, nTest, seed)
        self.r1 = np.sqrt(0.8-2/np.pi)
        self.r2 = np.sqrt(0.8)
        c1 = (self.ds[:, 0]**2 + self.ds[:, 1]**2 < self.r1**2).type(torch.int64)
        c2 = (self.ds[:, 0]**2 + self.ds[:, 1]**2 < self.r2**2).type(torch.int64)
        self.label = c1 + c2

    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
        theta = np.linspace(0, 2*np.pi, 100)
        return [self.r1*np.cos(theta), self.r2*np.cos(theta)], [self.r1*np.sin(theta), self.r2*np.sin(theta)]
 

# 2D dataSet. Classification in 2 classes.
# The points below the curve -2x + 3/2*sin(pi x) are associated with the first 
# class, the others with the second one.
class NonConvex(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        DataSet.__init__(self, "non convex", 2, 2, nTrain, nValid, nTest, seed)
        self.label =  (self.ds[:, 1] < -2*self.ds[:, 0] + 3/2*np.sin(np.pi*self.ds[:, 0])).type(torch.int64) 

    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
        x = np.linspace(-1, 1, 100)
        return [x], [-2*x+3/2*np.sin(np.pi*x)]


# 2D dataSet. Classification in 2 classes.
# The points inside the annulus of radii (0.8-2/pi)^(1/2) and (2/pi)^(1/2) are
# assigned to the first class, the others to the second one.
class BinaryAnnulus(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        self.r1 = np.sqrt(0.8-2/np.pi)
        self.r2 = np.sqrt(0.8)
        DataSet.__init__(self, "binary annulus", 2, 2, nTrain, nValid, nTest, seed)
        self.label = ((self.ds[:, 0]**2 + self.ds[:, 1]**2 < self.r2**2) & (self.ds[:, 0]**2 + self.ds[:, 1]**2 >  self.r1**2)).type(torch.int64) 

    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
            theta = np.linspace(0, 2*np.pi, 100)
            return [self.r1*np.cos(theta), self.r2*np.cos(theta)], [self.r1*np.sin(theta), self.r2*np.sin(theta)]


# 3D dataSet. Classification in 2 classes.
# The points inside the sphere of radius (3/pi)^(1/3) are associated with the 
# first class, the others with the second one.
class Sphere(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        DataSet.__init__(self, "sphere", 3, 2, nTrain, nValid, nTest, seed)
        self.label =  (self.ds[:, 0]**2 + self.ds[:, 1]**2 + self.ds[:, 2]**2 < np.cbrt(3/np.pi)**2).type(torch.int64) 


# 2D dataSet. Classification in 4 classes.
# The points are assigned to one for four classes depending on the sign of 
# the coordinates. The boundaries between the four regions are the x and y axes
class Squares(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        DataSet.__init__(self, "squares", 2, 4, nTrain, nValid, nTest, seed)
        s1 = ((self.ds[:, 0] > 0) & (self.ds[:, 1] > 0)).type(torch.int64)
        s2 = ((self.ds[:, 0] > 0) & (self.ds[:, 1] < 0)).type(torch.int64)
        s3 = ((self.ds[:, 0] < 0) & (self.ds[:, 1] > 0)).type(torch.int64)
        self.label = 2*s1+s2+3*s3

    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
        dom = np.linspace(-1, 1, 2)
        return [dom, 0*dom],[0*dom, dom] 


# 2D dataSet. Classification in 4 classes.
# The curves sin(pi x) +- x split the domain in four regions, assigned to the 
# four different classes.
class WavyLines(DataSet):

    def __init__(self, nTrain=5000, nValid=1000, nTest=1000, seed = None):
        DataSet.__init__(self, "wavy lines", 2, 4, nTrain, nValid, nTest, seed)
        w1 = (self.ds[:, 1] < np.sin(np.pi*self.ds[:, 0])+self.ds[:, 0]).type(torch.int64)
        w2 = (self.ds[:, 1] < np.sin(np.pi*self.ds[:, 0])-self.ds[:, 0]).type(torch.int64)
        self.label = w1+2*w2

    # Function to get a representation of the boundaries between two classes (used in plotting)
    def getShape(self):
        dom = np.linspace(-1, 1, 100)
        return [dom, dom], [np.sin(dom*np.pi)+dom, np.sin(dom*np.pi)-dom]

# --------------------

# Utility function to plot all the implemented 2D dataSets 
def plot2D():
    f=plt.figure(figsize=(12,6))
    gs = f.add_gridspec(2, 4)

    f.add_subplot(gs[0,0])
    ds = Circle()
    ds.plot()

    f.add_subplot(gs[0,1])
    ds = ThreeCircles()
    ds.plot()

    f.add_subplot(gs[0,2])
    ds = Annulus()
    ds.plot()

    f.add_subplot(gs[0,3])
    ds = NonConvex()
    ds.plot()

    f.add_subplot(gs[1,0])
    ds = BinaryAnnulus()
    ds.plot()
    
    f.add_subplot(gs[1,1])
    ds = Squares()
    ds.plot()

    f.add_subplot(gs[1,2])
    ds = WavyLines()
    ds.plot()

    plt.tight_layout()
    plt.show()
