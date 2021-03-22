# CSS-485 Final Project

**Note:** The folder `d_transfer_functions` stores the derivatives of the functions in `transfer_functions`. Likewise, the folder `d_cost_functions` stores the derivatives of the functions in `cost_functions`.

## 3/12/2021

**Added**  learning rate decay. You can specify max and min learning rate to perform linear learning rate decay based on epoch progress. If you want a constant learning rate, pass the same number for max and min learning rate.

Example:
```
lr_max = 1;
lr_min = 0.02;

...

mlp.add_layer(PerceptronLayer(350, 784, @sigmoid, @d_sigmoid, lr_max, lr_min, momentum, std));
mlp.add_layer(PerceptronLayer(170, 350, @sigmoid, @d_sigmoid, lr_max, lr_min,momentum, std));
mlp.add_layer(PerceptronLayer(10, 170, @relu, @d_relu, lr_max, lr_min, momentum, std));
```

## 3/11/2021

**Added**  models directory. Trained models are saved here. 

**Added**  Ensemble main. The file `ensemble_main.m` now uses saved models instead of training new ones in it.

Example:
```
% load models
load("models/model_2021_3_3_5_14.mat", "mlp");     
ensemble.add_model(mlp);

load("models/aug_model_2021_3_4_5_9.mat", "mlp");     
ensemble.add_model(mlp);
```

## 3/10/2021

**Added**  Ensemble model. This model trains multiple independent models and has them vote on a final prediction.

The file `ensemble_main.m` contains everything needed to do a full run with an Ensemble model.

Example:
```
ensemble = Ensemble();
    
% build models
for i = 1:3
    mlp = MultilayerPerceptron(@cross_entropy, @d_cross_entropy);

    mlp.add_layer(PerceptronLayer(20, 784, @my_tanh, @d_my_tanh, lr, momentum, std));
    mlp.add_layer(PerceptronLayer(10, 20, @softmax, @d_softmax, lr, momentum, std));
    
    ensemble.add_model(mlp);
end


% train the model
[losses, acc, acc_list] = ensemble.fit( ...
    train_data, ...
    train_labels, ...
    epochs, ...
    batch_size, ...
    stop_buff, ...
    stop_thresh, ...
    test_data, ...
    test_labels, ...
    @accuracy ...
);
```

**Added** model weights for ensemble voting in the function `utility/compute_prec_rec_weights.m`. Each model has a weight vector consisting of a weight value for each individual class (10 classes in this case). This value is computed by `precision * recall`. It will approach 1 if both precision and recall are high, and approach 0 if one or both are low.

```
validation_preds = hardmax(mlp.frozen_forward(test_set));
validation_labels % already set

weights = compute_prec_rec_weights(validation_preds, validation_labels);
```
```
weights = 
    0.6200  % (precision * recall) for 0 class
    0.8630  % (precision * recall) for 1 class
    0.4916  % (precision * recall) for 2 class
    0.6950  % ... and so on
    0.5086
    0.8238
    0.3044
    0.7788
    0.8577
    0.8261
```

## 3/5/2021

**Added** generate_agumented_data.py
it performs data augmentation on the train set

**Added** main_augmented.m 
this should be used to train the model with augmented data created with generate_augmented_data.py


## 3/3/2021

**Added** normalization to submission data.

**Added** cross-entropy loss and softmax transfer function. Softmax should be used on the output layer and should be paired with cross-entropy loss.

Example:

```
mlp = MultilayerPerceptron(@cross_entropy, @d_cross_entropy);

mlp.add_layer(PerceptronLayer(256, 784, @my_tanh, @d_my_tanh, lr, momentum, std));
mlp.add_layer(PerceptronLayer(64, 256, @relu, @d_relu, lr, momentum, std));
mlp.add_layer(PerceptronLayer(10, 64, @softmax, @d_linear, lr, momentum, std));
```

Note: The folder `d_transfer_functions` stores the derivatives of the functions in `transfer_functions`. Likewise, the folder `d_cost_functions` stores the derivatives of the functions in `cost_functions`.
