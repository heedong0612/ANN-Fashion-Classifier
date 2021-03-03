# CSS-485 Final Project

**Note:** The folder `d_transfer_functions` stores the derivatives of the functions in `transfer_functions`. Likewise, the folder `d_cost_functions` stores the derivatives of the functions in `cost_functions`.

### 3/3/2021

**Added** normalization to submission data.


**Added** cross-entropy loss and softmax transfer function. Softmax should be used on the output layer and should be paired with cross-entropy loss.

Example:

```
mlp = MultilayerPerceptron(@cross_entropy, @d_cross_entropy);

mlp.add_layer(PerceptronLayer(256, 784, @my_tanh, @d_my_tanh, lr, momentum, std));
mlp.add_layer(PerceptronLayer(256, 784, @relu, @d_relu, lr, momentum, std));
mlp.add_layer(PerceptronLayer(10, 256, @softmax, @d_softmax, lr, momentum, std));
```

