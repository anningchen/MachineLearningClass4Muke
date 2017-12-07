
# coding: utf-8

# In[ ]:




from numpy import *

class SimplePerceptron():
    """ A basic Perceptron Class for Machine Learning 1st"""
    
    def __init__(self,inputs,targets):
        """ Constructor """
        # 初始设置,首选获取输入X的维度
        if ndim(inputs)>1:
            self.nIn = shape(inputs)[1]
        else: 
            self.nIn = 1

        if ndim(targets)>1:
            self.nOut = shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = shape(inputs)[0]

        # 初始化权重,随机数, +1的原因是为了多了偏置b
        self.weights = random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self,inputs,targets,eta,nIterations):
        """ 模型训练过程,就是权重W的更新过程 """
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.nData,1))),axis=1)
    
        # Training
        change = range(self.nData)

        for n in range(nIterations):

            self.outputs = self.pcnfwd(inputs);
            self.weights += eta*dot(transpose(inputs),targets-self.outputs)
            print("Iteration: ", n)
            print(self.weights)

            activations = self.pcnfwd(inputs)
            print("Final outputs are:")
            print(activations)

    def pcnfwd(self,inputs):
    #前向传输
    #/Y=XW
        outputs =  dot(inputs,self.weights)

        # 给输出Y一个阈值,大于0输出1,小于0输出0
        """if outputs>0
                outpust=1
            else 
                ouputs=0"""
        return where(outputs>0,1,0)

