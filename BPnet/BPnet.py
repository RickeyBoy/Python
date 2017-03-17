import numpy as np
from pylab import *
import matplotlib.pyplot as plt


def Sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def Sigmoid_output_to_derivative(output):
    return Sigmoid(output)*(1-Sigmoid(output))

class InputLayer:
    def __init__(self, number):
        self.number = number
        self.sampleinput = []
        self.u = []
        self.v = []
        self.b = []
    
    def SetInput(self, training_set_input): #training_set_input is a column matrix
        self.sampleinput = training_set_input


class HiddenLayer:
    def __init__(self, number, InputLayer, OutputLayer, LearningRate):
        self.number = number
        self.inputlayer = InputLayer
        self.outputlayer = OutputLayer
        
        self.u = []
        self.v = []
        self.b = []
        self.SetInputLayer()
        self.SetOutputLayer()
        self.learningrate = LearningRate
        self.net = []
        self.output = []
        self.LG = []  #LG means Local Gradient
        for i in range(number):
            self.net.append(0)
            self.output.append(0)
            self.LG.append(0)
    def GetInputLayer(self):
        return self.inputlayer

    def GetOutputLayer(self):
        return self.outputlayer

    def SetInputLayer(self):
        tmp = []
        for j in range(self.number):
            for i in range(self.inputlayer.number):
                tmp.append(0.1*j)
            self.inputlayer.u.append(np.matrix(tmp))
            self.inputlayer.v.append(np.matrix(tmp))
            self.inputlayer.b.append(0.1*j)
            tmp = []

    def SetOutputLayer(self):
        tmp = []
        for k in range(self.outputlayer.number):
            for j in range(self.number):
                tmp.append(0.1*j)
            self.u.append(np.matrix(tmp))
            self.v.append(np.matrix(tmp))
            self.b.append(0)
            tmp = []
    
    def Calculate(self):
        for j in range(self.number):
            self.net[j] = (self.inputlayer.u[j]*np.multiply(self.inputlayer.sampleinput,self.inputlayer.sampleinput)).A[0][0] + (self.inputlayer.v[j]*self.inputlayer.sampleinput).A[0][0] + self.inputlayer.b[j]
            self.output[j] = Sigmoid(self.net[j])
        tmp = np.matrix(self.output).T
        for k in range(self.outputlayer.number):
            self.outputlayer.net[k] = (self.u[k]*np.multiply(tmp, tmp)).A[0][0]+(self.v[k]*tmp).A[0][0]+self.b[k]
            self.outputlayer.output[k] = Sigmoid(self.outputlayer.net[k])
            self.outputlayer.e[k] = self.outputlayer.sampleoutput[k] - self.outputlayer.output[k]
            self.outputlayer.LG[k] = self.outputlayer.e[k]*Sigmoid_output_to_derivative(self.outputlayer.net[k])
        for j in range(self.number):
            tmp = 0
            for k in range(self.outputlayer.number):
                tmp = tmp + self.outputlayer.LG[k]*((2*self.u[k]).A[0][j]*self.output[j]+(self.v[k]).A[0][j])
            self.LG[j] = tmp*Sigmoid_output_to_derivative(self.net[j])

    def Error(self):
        result = 0
        for k in range(self.outputlayer.number):
            result = result + self.outputlayer.e[k]**2
        return result/2.0

    def Learn(self):
        self.Calculate()
        for k in range(self.outputlayer.number):
            tmp = np.matrix(self.output)
            self.u[k] = self.u[k] + self.learningrate*self.outputlayer.LG[k]*np.multiply(tmp, tmp)
            self.v[k] = self.v[k] + self.learningrate*self.outputlayer.LG[k]*tmp
            self.b[k] = self.b[k] + self.learningrate*self.outputlayer.LG[k]
        for j in range(self.number):
            tmp = self.inputlayer.sampleinput.T
            self.inputlayer.u[j] = self.inputlayer.u[j] + self.learningrate*self.LG[j]*np.multiply(tmp, tmp)
            self.inputlayer.v[j] = self.inputlayer.v[j] + self.learningrate*self.LG[j]*tmp
            self.inputlayer.b[j] = self.inputlayer.b[k] + self.learningrate*self.LG[j]

    def Check(self, inputMatrix): #return output array
        self.inputlayer.SetInput(inputMatrix)
        for j in range(self.number):
            self.net[j] = (self.inputlayer.u[j]*np.multiply(self.inputlayer.sampleinput,self.inputlayer.sampleinput)).A[0][0] + (self.inputlayer.v[j]*self.inputlayer.sampleinput).A[0][0] + self.inputlayer.b[j]
            self.output[j] = Sigmoid(self.net[j])
        tmp = np.matrix(self.output).T
        for k in range(self.outputlayer.number):
            self.outputlayer.net[k] = (self.u[k]*np.multiply(tmp, tmp)).A[0][0]+(self.v[k]*tmp).A[0][0]+self.b[k]
            self.outputlayer.output[k] = Sigmoid(self.outputlayer.net[k])
        return self.outputlayer.output


class OutputLayer:
    def __init__(self, number):
        self.number = number
        self.sampleoutput = []
        self.net = []
        self.output = []
        self.e = []
        self.LG = []
        for i in range(number):
            self.net.append(0)
            self.output.append(0)
            self.e.append(0)
            self.LG.append(0) #LG means Local Gradient
    
    def SetOutput(self, training_set_output):
        self.sampleoutput = training_set_output


training_set_input=[]
training_set_output=[]

test_set_input=[]
test_set_output=[]

def main(iteration):
    global training_set_input, training_set_output
    training_set_input=[]
    training_set_output=[]
    test_set_input=[]
    test_set_output=[]
    filein = open('./data/two_spiral_test.txt','r')
    filein2 = open('./data/two_spiral_test.txt','r')
    for line in filein.readlines():
        InputSample = []
        tmp = line.split('\t')
        InputSample.append(eval(tmp[0]))
        InputSample.append(eval(tmp[1]))
        training_set_input.append(np.matrix(InputSample).T)
        training_set_output.append(eval(tmp[2]))
        
    for line in filein2.readlines():
        InputSample = []
        tmp = line.split('\t')
        InputSample.append(eval(tmp[0]))
        InputSample.append(eval(tmp[1]))
        test_set_input.append(np.matrix(InputSample).T)
        test_set_output.append(eval(tmp[2]))
    
    Layer1 = InputLayer(2)
    Layer3 = OutputLayer(1)
    Layer2 = HiddenLayer(10, Layer1, Layer3, 0.5)
    length = len(training_set_input)
    length2 = len(test_set_input)
    best_count = 0
    best_rate = 2
    for i in range(iteration):
        Layer1.SetInput(training_set_input[i%length])
        Layer3.SetOutput([training_set_output[i%length]])
        Layer2.Learn()
        if(i%100==0):
            count = 0
            count2 = 0
            for j in range(length):
                if(Layer2.Check(training_set_input[j])[0]>0.5):
                    if(training_set_output[j]==0):
                        count = count+1
                else:
                    if(training_set_output[j]==1):
                        count = count+1
            for j in range(length2):
                if(Layer2.Check(test_set_input[j])[0]>0.5):
                    if(test_set_output[j]==0):
                        count2 = count2+1
                else:
                    if(test_set_output[j]==1):
                        count2 = count+2
            print "at iteration", i, "False_rate1=",count*1.0/length,"False_rate2=",count2*1.0/length2
            if(count*1.0/length+count2*1.0/length2<best_rate):
                best_rate = count*1.0/length+count2*1.0/length2
                best_count = i
    print "best_rate=",best_rate,"best_count",best_count
    draw(Layer2)

def draw(Layer2):
    ax = gca()
    xlim(-4,4)
    xticks(np.linspace(-4,4,9,endpoint=True))
    ylim(-4,4)
    yticks(np.linspace(-4,4,9,endpoint=True))

    X = np.linspace(-4,4,101,endpoint=True)
    Y = np.linspace(-4,4,101,endpoint=True)

    X0 = []
    Y0 = []
    X1 = []
    Y1 = []

    for i in range(101):
        for j in range(101):
            if(Layer2.Check(np.matrix([X[i],Y[j]]).T)[0]>0.5):
                X1.append(X[i])
                Y1.append(Y[j])
            else:
                X0.append(X[i])
                Y0.append(Y[j])
    X0 = np.array(X0)
    Y0 = np.array(Y0)
    X1 = np.array(X1)
    Y1 = np.array(Y1)


    #scatter(X0,Y0,s = 50,c='blue',linewidths = 0)
    #scatter(X1,Y1,s = 50,c='red',linewidths = 0)

    X2 = [] #white, 0
    Y2 = []
    X3 = [] #black, 1
    Y3 = []

    for i in range(len(training_set_input)):
        if(training_set_output[i]):
            X3.append(training_set_input[i].A[0][0])
            Y3.append(training_set_input[i].A[1][0])
        else:
            X2.append(training_set_input[i].A[0][0])
            Y2.append(training_set_input[i].A[1][0])

    X2 = np.array(X2)
    Y2 = np.array(Y2)
    X3 = np.array(X3)
    Y3 = np.array(Y3)

    #scatter(X2,Y2,s = 50,c='white')
    #scatter(X3,Y3,s = 50,c='black')
    #show()



    plt.plot(X0,Y0,'ro',alpha = 0.1)
    plt.plot(X1,Y1,'bo',alpha = 0.1)
    plt.plot(X2,Y2,'ro')
    plt.plot(X3,Y3,'bo')
    plt.show()
            
if __name__ == "__main__":
    main(100000)