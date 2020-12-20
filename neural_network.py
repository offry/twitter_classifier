from defines import *


def create_network(X_train_shape):
    input_dim = X_train_shape  # Number of features
    first_layer_out_shape = int(2*X_train_shape/3)
    second_layer_out_shape = int(X_train_shape / 3)
    model = Sequential()
    model.add(layers.Dense(first_layer_out_shape, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(second_layer_out_shape, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])
    model.summary()

    checkpoint_path = "neural_networks/cp.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return model, cp_callback

def train_net(model, X_train, Y_train, cp_callback,new_fit = 0):
    if new_fit:
        model.fit(X_train, Y_train, epochs = 10,batch_size = 10, callbacks=[cp_callback])
    else:
        model.load_weights("neural_networks/cp.ckpt")
    return model

def evaluate_net(model,X_test, Y_test):
    _, accuracy = model.evaluate(X_test, Y_test)
    print("results of classification by neural network with word to vec:")
    print('Accuracy: %.2f' % (accuracy * 100))