# import libraries
from __future__ import print_function
import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import models
import optuna
import tensorflow
from tensorboard.plugins.hparams import api as hp

# batch, classes, epochs
batch_size = 32
num_classes = 10
epochs = 50 # 50

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# plotting some random 10 images
class_names = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:] == i)[0]
    features_idx = x_train[idx, ::]
    img_num = np.random.randint(features_idx.shape[0])
    im = (features_idx[img_num, ::])
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# convert to float, normalise the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# train 
def train_valid(model, state):
    
    log_dir = f"logs/fit_{study.study_name}/trial-{state}"
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history=model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True, callbacks=[tensorboard_callback])

    val_acc=max(history.history['val_accuracy'])
    return val_acc
   

def objective(trial):
    
    params = {
            'num_filters_1':trial.suggest_int('num_filters_1',16,32,8),
            'num_filters_2':trial.suggest_int('num_filters_2',32,64,16),
            'num_filters_3':trial.suggest_int('num_filters_3',64,128,32),
            'size': trial.suggest_int('size',2,4,1),
            'p_size': trial.suggest_int('p_size',2,3),
            'L_ReLU_alpha': trial.suggest_float('L_ReLU_alpha', 0, 0.3,step=0.05),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop","sgd"]),
            'dropout_rate': trial.suggest_float("dropout_rate", 0.1, 0.3),
            'd_lay': trial.suggest_int('d_lay',256,1024,256)
              } 
       
    model=models.build_model(study.study_name ,x_train,num_classes,params)
    accuracy=train_valid(model, trial.number)

    with open(f'tmp/{study.study_name}_trial-{trial.number}.txt','w') as file:
        file.write(f'{study.study_name}')
        file.write(f'number of filters 1: {params["num_filters_1"]}\n')
        file.write(f'number of filters 2: {params["num_filters_2"]}\n')
        file.write(f'number of filters 3: {params["num_filters_3"]}\n')
        file.write(f'size filters: {params["size"]}\n')
        file.write(f'Max Pooling 2D size: {params["p_size"]}\n')
        file.write(f'activation function: LeakyReLU with alpha= {params["L_ReLU_alpha"]}\n')
        file.write(f'dropout={params["dropout_rate"]}\n')
        file.write(f'dense layer={params["d_lay"]}\n')
        file.write(f'optimizer: {params["optimizer"]}\n')
        file.write(f'accuracy={accuracy}')
        file.write('\n')
        file.close()    
    #for table Hparams in tensorboard 
    with tensorflow.summary.create_file_writer(f'logs/fit/hparam_tuning/{study.study_name}/trial-{trial.number}').as_default():
      hp.hparams(params)
      tensorflow.summary.scalar('accuracy', accuracy, step=1)
    return accuracy

def run():
    with open(f'tmp/best_params.txt','w') as file:
        for i in ['Model_1','Model_2','Model_3','Model_4','Model_5','Model_6','Model_7']:
            global study
            study = optuna.create_study(study_name=i,direction="maximize", sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=15)
            file.write(f'best trial in model-{study.study_name}: trial-{study.best_trial._number}\n')
        
if __name__== "__main__":
    run()
    #tensorboard dev upload --logdir ./logs --name "Simple experiment with GRU and LSTM" --description "?"