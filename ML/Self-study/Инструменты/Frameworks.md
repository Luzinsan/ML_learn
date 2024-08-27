# Layers

| Name                       | Pytorch             | Mxnet | Jax   | Tensorflow |
| -------------------------- | ------------------- | ----- | ----- | ---------- |
| Линейный полносвязный слой | LazyLinear (Linear) | Dense | Dense | Dense      |

# Loss

| Name                  | Pytorch       | Mxnet                   | Jax                                       | Tensorflow                    |
| --------------------- | ------------- | ----------------------- | ----------------------------------------- | ----------------------------- |
| L2                    | MSELoss       | L2Loss                  | l2_loss                                   | MeanSquaredError              |
| Softmax Cross Entropy | cross_entropy | SoftmaxCrossEntropyLoss | softmax_cross_entropy_with_integer_labels | SparseCategoricalCrossentropy |
# Optimizers

| Name                             | Pytorch | Mxnet | Jax | Tensorflow |
| -------------------------------- | ------- | ----- | --- | ---------- |
| Стохастический градиентный спуск | SGD     | sgd   | sgd | SGD        |
# Activation Functions

| Name | Pytorch   | Mxnet                             | Jax       | Tensorflow                        |
| ---- | --------- | --------------------------------- | --------- | --------------------------------- |
| ReLU | nn.ReLU() | Параметр в слое activation='relu' | nn.relu() | Параметр в слое activation='relu' |
