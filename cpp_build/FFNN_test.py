import FFNN_pymodule
import numpy as np
import pickle
import json




def MakeTrainingData(num_samples):
    x_data=[]

    y_data=[]
    for i in range(num_samples):
        num1=np.random.randint(0,2)
        num2=np.random.randint(0,2)

        x_data.append([num1,num2])
        y_data.append([num1 ^ num2])
    
    return [x_data, y_data]



[x_data, y_data]=MakeTrainingData(2000)

model = FFNN_pymodule.FFNN_Builder()

model.setTopology([2, 4, 1])
model.setActivationFunction('sigmoid')
model.setEpochs(20)

model.fitModel(x_data, y_data)

print("test model [1,0]")
print(model.predict([1, 0]))

'''

pickle.dump(model,open('model_file.p',"wb"), pickle.HIGHEST_PROTOCOL)
model2=pickle.load(open('model_file.p','rb'))
print('now from pickle')

print("test model [1,0]")
print(model2.predict([1, 0]))
'''

