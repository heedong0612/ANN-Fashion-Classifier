# CSS-485 Final Project

**Note:** The folder `d_transfer_functions` stores the derivatives of the functions in `transfer_functions`. Likewise, the folder `d_cost_functions` stores the derivatives of the functions in `cost_functions`.

### 3/10/2021

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

### 3/3/2021

**Added** normalization to submission data.


**Added** cross-entropy loss and softmax transfer function. Softmax should be used on the output layer and should be paired with cross-entropy loss.

Example:

```
mlp = MultilayerPerceptron(@cross_entropy, @d_cross_entropy);

mlp.add_layer(PerceptronLayer(256, 784, @my_tanh, @d_my_tanh, lr, momentum, std));
mlp.add_layer(PerceptronLayer(64, 256, @relu, @d_relu, lr, momentum, std));
mlp.add_layer(PerceptronLayer(10, 64, @softmax, @d_linear, lr, momentum, std));
```

