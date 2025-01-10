# Simple Neural Network Implementation in Python

This repository contains a simple implementation of a neural network using Python and NumPy. The network has:

- **2 input features**
- **1 hidden layer** with 2 neurons
- **1 output neuron**

The network is trained using a custom implementation of backpropagation and gradient descent. This is intended as an educational project to demonstrate the inner workings of a basic neural network.

---

## Features

1. **Feedforward mechanism**: Calculates predictions based on input data.
2. **Backpropagation**: Updates weights and biases using gradient descent.
3. **Custom loss function**: Uses Mean Squared Error (MSE) for loss calculation.
4. **Sigmoid activation function**: Applied at each neuron to introduce non-linearity.
5. **Educational focus**: Demonstrates the basic principles of neural network operation.

---

## Requirements

To run this project, you need Python 3.x and NumPy installed. You can install NumPy using pip:

```bash
pip install numpy
```

---

## Code Structure

### Functions

- **`sigmoid(x)`**: Calculates the sigmoid activation for a given input.
- **`deriv_sigmoid(x)`**: Computes the derivative of the sigmoid function.
- **`mse_loss(y_true, y_pred)`**: Calculates the Mean Squared Error loss.

### `OurNeuralNetwork` Class

- **Initialization**: Randomly initializes weights and biases.
- **`feedforward(x)`**: Performs a forward pass through the network.
- **`train(data, all_y_trues)`**: Trains the network using backpropagation.

---

## Dataset

The network is trained on a small dataset:

| Feature 1 | Feature 2 | Label |
|-----------|-----------|-------|
| -2        | -1        | 1     |
| 25        | 6         | 0     |
| 17        | 4         | 0     |
| -15       | -6        | 1     |

The labels are binary, with `1` representing one class (e.g., female) and `0` representing the other class (e.g., male).

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Vatsa10/Neural-Networks-From-Scratch
   cd Neural-Networks-From-Scratch
   ```

2. Run the script:
   ```bash
   python neural.py
   ```

3. During training, the loss for each epoch will be printed every 10 epochs.  
   Example output:
   ```
   Epoch 0 loss: 0.295
   Epoch 10 loss: 0.202
   ...
   Epoch 990 loss: 0.005
   ```

4. After training, you can make predictions using the `feedforward` method:
   ```python
   emily = np.array([-7, -3])  # Example input
   prediction = network.feedforward(emily)
   print("Prediction:", prediction)
   ```

---

## Example Output

After training, the network predicts the following probabilities:

- Emily (`[-7, -3]`): `~0.951` (close to 1, likely female)
- Frank (`[20, 2]`): `~0.039` (close to 0, likely male)

---

## Limitations

- This implementation is for educational purposes and is not optimized for performance.
- The dataset is extremely small and only serves as a demonstration.
- The network may not generalize well to other datasets without significant modification.

---

## Contributing

Contributions are welcome! If you have ideas to improve this implementation or want to add features, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This implementation is inspired by simple neural network tutorials and aims to provide a step-by-step guide for beginners in machine learning and deep learning.

---

Let me know if you need any changes!
