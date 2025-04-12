//! # Handwritten Digit Recognizer (MNIST) in Rust
//!
//! This is a simple implementation of a 2-layer neural network for classifying handwritten digits from the MNIST dataset.
//!
//! - Input Layer: 784 units (28x28 pixels)
//! - Hidden Layer: `neurons` units with ReLU activation
//! - Output Layer: 10 units with Softmax activation
//!
//! ## Mathematical Formulation
//!
//! ### Forward Propagation
//! Let \( X \in \mathbb{R}^{784 \times m} \) be the input matrix (m samples).
//!
//! - Hidden pre-activation: \( Z_1 = W_1 X + b_1 \)
//! - Hidden activation: \( A_1 = \text{ReLU}(Z_1) \)
//! - Output pre-activation: \( Z_2 = W_2 A_1 + b_2 \)
//! - Output activation: \( A_2 = \text{Softmax}(Z_2) \)
//!
//! ### Backward Propagation
//! Gradient descent is used to minimize the cross-entropy loss.

use std::{f64, iter::zip};

use clap::Parser;
use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr, s};
use plotters::prelude::*;
use polars::{io::SerReader, prelude::*};
use rand::{Rng, distr::Uniform, random, seq::SliceRandom};

/// Alias for a 2D array of f64
type Array2d = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

/// Alias for a 1D array of f64
type Array1d = ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;

/// Command line arguments
#[derive(Parser, Debug)]
struct Args {
    /// Set how much iteration to train
    #[arg(short, long)]
    iters: u32,

    /// Set neurons par hidden layer
    #[arg(short, long)]
    neurons: usize,

    /// Set how many time to test the trained neural network
    #[arg(short, long)]
    test_prediction: usize,

    /// Set the Learning rate α
    #[arg(short, long)]
    alpha: f64,
}

/// Entry point: loads MNIST data, trains a neural network using gradient descent,
/// and visualizes predictions on the test set.
fn main() {
    // ------------------------
    // 1. Parse CLI Arguments
    // ------------------------
    let args = Args::parse();

    // Initialize random number generator
    let mut rng = rand::rng();

    // ------------------------
    // 2. Load and Preprocess Data
    // ------------------------

    // Load CSV data
    let csv_data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/train.csv".into()))
        .unwrap()
        .finish()
        .unwrap();

    // Convert CSV data to ndarray
    let data = csv_data.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    let (m, n) = data.dim();

    // Shuffle rows to prevent overfitting to dataset order
    let mut rows: Vec<_> = data.axis_iter(Axis(0)).collect();
    rows.shuffle(&mut rng);

    // Stack shuffled rows back into a single ndarray
    let data = ndarray::stack(
        Axis(0),
        &rows.iter().map(|row| row.view()).collect::<Vec<_>>(),
    )
    .unwrap();

    // Split into dev and train sets

    // Slice the first 1000 columns from the original dataset for the development set (dev set).
    let data_dev_slice = data.slice(s![0..1000, ..]);
    // Transpose the dev set so that the rows represent individual samples (column-major format).
    let data_dev = data_dev_slice.t(); // shape: (n, 1000) => transpose for column-major

    // The first row of the transposed data is the target labels (y_dev) for the development set.
    let y_dev = data_dev.slice(s![0, ..]);
    // The remaining rows are the features (x_dev) for the development set.
    let x_dev_slice = data_dev.slice(s![1..n, ..]);
    // Normalize the feature values by dividing by 255.0 to scale the values between 0 and 1.
    let x_dev = x_dev_slice.to_owned() / 255.0;

    // Slice the remaining rows from the original dataset for the training set (train set).
    let data_train_slice = data.slice(s![1000..m, ..]);
    // Transpose the training set for column-major format.
    let data_train = data_train_slice.t();

    // The first row of the transposed data is the target labels (y_train) for the training set.
    let y_train = data_train.row(0);
    // The remaining rows are the features (x_train) for the training set.
    let x_train_slice = data_train.slice(s![1..n, ..]);
    // Normalize the feature values by dividing by 255.0 to scale the values between 0 and 1.
    let x_train = x_train_slice.to_owned() / 255.0;

    // ------------------------
    // 3. Train Neural Network
    // ------------------------
    let (w1, b1, w2, b2) = gradient_descent(
        x_train.to_owned(),
        y_train.to_owned(),
        args.iters,
        args.alpha,
        args.neurons,
    )
    .into();

    // ------------------------
    // 4. Evaluate with Predictions
    // ------------------------

    for _ in 0..args.test_prediction {
        test_prediction(
            random::<u8>() as usize,
            &x_dev,
            &y_dev.to_owned(),
            &w1,
            &b1,
            &w2,
            &b2,
        );
    }
}

/// Initializes the parameters (weights and biases) for both layers of the neural network.
///
/// # Returns
/// Returns an array of 4 2d arrays:
/// - \( W_1 \in \mathbb{R}^{h \times 784} \): weights from input to hidden layer
/// - \( b_1 \in \mathbb{R}^{h \times 1} \): biases for hidden layer
/// - \( W_2 \in \mathbb{R}^{10 \times h} \): weights from hidden to output layer
/// - \( b_2 \in \mathbb{R}^{10 \times 1} \): biases for output layer
fn init_params(neurons: usize) -> [Array2d; 4] {
    let mut rng = rand::rng();
    let dist = Uniform::new(0., 1.).unwrap();

    // Initialize weights and biases with random values in [-0.5, 0.5)
    // \( W_1 \): Weights for the input to hidden layer, shape \( (h \times 784) \)
    let w1 = Array2::from_shape_fn((neurons, 784), |_| rng.sample(dist)) - 0.5; // Random values between -0.5 and 0.5
    // \( b_1 \): Biases for the hidden layer, shape \( (h \times 1) \)
    let b1 = Array2::from_shape_fn((neurons, 1), |_| rng.sample(dist)) - 0.5; // Random values between -0.5 and 0.5

    // \( W_2 \): Weights for the hidden to output layer, shape \( (10 \times h) \)
    let w2 = Array2::from_shape_fn((10, neurons), |_| rng.sample(dist)) - 0.5; // Random values between -0.5 and 0.5
    // \( b_2 \): Biases for the output layer, shape \( (10 \times 1) \)
    let b2 = Array2::from_shape_fn((10, 1), |_| rng.sample(dist)) - 0.5; // Random values between -0.5 and 0.5

    // Return the initialized weights and biases as an array of 4 elements
    [w1, b1, w2, b2]
}

/// Applies the ReLU activation function element-wise.
///
/// \[ \text{ReLU}(x) = \max(0, x) \]
fn relu(z: &Array2d) -> Array2d {
    let mut z = z.clone();
    // Apply ReLU element-wise: \( \text{ReLU}(x) = \max(0, x) \)
    z.par_mapv_inplace(|v| v.max(0.0)); // If v > 0, it stays; otherwise, it becomes 0.
    z
}

/// Computes the derivative of ReLU for backpropagation.
///
/// \[ \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases} \]
fn deriv_relu(z: &Array2d) -> Array2d {
    // Derivative of ReLU: 1 if x > 0, otherwise 0.
    z.mapv(|v| if v > 0. { 1. } else { 0. })
}

/// Applies the softmax function across the output layer.
/// Ensures output probabilities sum to 1 for each column (sample).
///
/// \[ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \]
fn softmax(z: &Array2d) -> Array2d {
    // Find the max value for each column to avoid overflow during exponentiation.
    let max_per_col = z.map_axis(Axis(0), |col| {
        col.iter().cloned().fold(f64::NEG_INFINITY, f64::max) // Get the max per column
    });
    // Shift the values in z to prevent overflow during exponentiation
    let shifted = z - &max_per_col.insert_axis(Axis(0)); // Subtract the max for each column
    // Exponentiate each element: \( e^{z_i} \)
    let exp = shifted.mapv(|x| x.exp()); // Apply exponential function element-wise
    // Sum the exponentiated values along each column: \( \sum_j e^{z_j} \)
    let sum_exp = exp.sum_axis(Axis(0)).insert_axis(Axis(0)); // Sum across columns
    // Return the normalized (probability) values: \( \frac{e^{z_i}}{\sum_j e^{z_j}} \)
    &exp / &sum_exp
}

/// Performs forward propagation for the neural network.
///
/// # Arguments
/// - \( W_1 \): Weights from input to hidden layer
/// - \( b_1 \): Biases for hidden layer
/// - \( W_2 \): Weights from hidden to output layer
/// - \( b_2 \): Biases for output layer
/// - \( X \): Input data (features)
///
/// # Returns
/// Returns the intermediate results during forward propagation:
/// - \( z_1 \): Linear combination at hidden layer
/// - \( a_1 \): Activations after ReLU at hidden layer
/// - \( z_2 \): Linear combination at output layer
/// - \( a_2 \): Activations after softmax at output layer
fn forward_prop(
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
    x: &Array2d,
) -> [Array2d; 4] {
    // Compute activations for the hidden layer:
    // \( z_1 = W_1 \cdot X + b_1 \)
    let z1 = w1.dot(x) + b1; // Matrix multiplication for weights and inputs, then add biases
    let a1 = relu(&z1); // Apply ReLU activation to z1: \( a_1 = \text{ReLU}(z_1) \)

    // Compute activations for the output layer:
    // \( z_2 = W_2 \cdot a_1 + b_2 \)
    let z2 = w2.dot(&a1) + b2; // Matrix multiplication for weights and hidden activations, then add biases
    let a2 = softmax(&z2); // Apply softmax activation to z2: \( a_2 = \text{Softmax}(z_2) \)

    // Return all intermediate results for use in backpropagation
    [z1, a1, z2, a2]
}

/// Converts labels into one-hot encoded matrix form.
/// Used for computing gradients and loss.
///
/// # Arguments
/// - \( y \): Array of labels (each label is an integer corresponding to the class)
///
/// # Returns
/// Returns the one-hot encoded matrix where each column represents the one-hot encoding of the label
/// for each sample.
fn one_hot(y: &Array1d) -> Array2d {
    // Find the maximum class label to determine the number of classes
    let max = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let num_samples = y.len();
    let num_classes = *max as usize + 1; // Assume classes are in range [0, max_class]
    let mut one_hot_y = Array2::<f64>::zeros((num_samples, num_classes)); // Initialize matrix of zeros

    // Set the appropriate index in each row to 1 based on the label
    for (i, class_idx) in y.iter().enumerate() {
        one_hot_y[[i, *class_idx as usize]] = 1.; // One-hot encoding: set the corresponding class to 1
    }

    // Return the transpose of the one-hot encoded matrix
    one_hot_y.t().to_owned() // Return transposed matrix to match expected shape
}

/// Performs the backpropagation step in a neural network, calculating the gradients
/// of weights and biases (dw1, db1, dw2, db2) based on the current activations
/// and the error of the network. This is a core part of the learning algorithm, allowing
/// the network to update its weights and biases to minimize the loss function.
///
/// # Parameters
/// - `z1`: The pre-activation values of the first layer (before the ReLU activation).
/// - `a1`: The activations of the first layer (after the ReLU activation).
/// - `a2`: The activations of the second layer (final output layer).
/// - `w2`: The weights for the second layer (connecting the first layer to the output).
/// - `x`: The input data.
/// - `y`: The true labels of the input data.
///
/// # Returns
/// Returns a tuple of the gradients for the weights and biases of both layers: `[dw1, db1, dw2, db2]`.
fn back_prop(
    z1: &Array2d,
    a1: &Array2d,
    // z2: Array2d, // Not used in this function but could be for the second layer's pre-activation.
    a2: &Array2d,
    w2: &Array2d,
    x: &Array2d,
    y: &Array1d,
) -> [Array2d; 4] {
    let m = y.len(); // The number of training examples.

    let one_hot_y = one_hot(y); // Converts the labels into one-hot encoding.

    // Compute the gradient of the loss with respect to the output layer activations (dz2).
    let dz2 = a2 - one_hot_y;

    // Compute the gradient of the loss with respect to the weights of the second layer (dw2).
    // This is the dot product of dz2 (error term) and the transpose of a1 (activations of the first layer).
    let dw2 = 1. / m as f64 * dz2.dot(&a1.t());

    // Compute the gradient of the loss with respect to the biases of the second layer (db2).
    // This is the sum of dz2 along axis 1, normalized by the number of examples (m).
    let db2 = dz2.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1. / m as f64);

    // Compute the gradient of the loss with respect to the first layer's activations (dz1).
    // This is the dot product of the transpose of w2 (weights of the second layer) and dz2 (error term),
    // followed by element-wise multiplication with the derivative of the ReLU function (deriv_relu).
    let dz1 = w2.t().dot(&dz2) * deriv_relu(z1);

    // Compute the gradient of the loss with respect to the weights of the first layer (dw1).
    // This is the dot product of dz1 (error term) and the transpose of x (input data).
    let dw1 = 1. / m as f64 * dz1.dot(&x.t());

    // Compute the gradient of the loss with respect to the biases of the first layer (db1).
    // This is the sum of dz1 along axis 1, normalized by the number of examples (m).
    let db1 = dz1.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1. / m as f64);

    // Return the gradients for the weights and biases of both layers.
    [dw1, db1, dw2, db2]
}

/// Updates the parameters (weights and biases) of the network using the gradients
/// calculated in backpropagation and a learning rate (alpha). This is the step
/// where the model learns by adjusting its weights and biases.
///
/// # Parameters
/// - `w1`: The current weights of the first layer.
/// - `b1`: The current biases of the first layer.
/// - `w2`: The current weights of the second layer.
/// - `b2`: The current biases of the second layer.
/// - `dw1`: The gradient of the loss with respect to the weights of the first layer.
/// - `db1`: The gradient of the loss with respect to the biases of the first layer.
/// - `dw2`: The gradient of the loss with respect to the weights of the second layer.
/// - `db2`: The gradient of the loss with respect to the biases of the second layer.
/// - `alpha`: The learning rate (how much the weights are adjusted during training).
///
/// # Returns
/// Returns the updated weights and biases `[w1, b1, w2, b2]`.
#[allow(clippy::too_many_arguments)]
fn update_params(
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
    dw1: &Array2d,
    db1: &Array2d,
    dw2: &Array2d,
    db2: &Array2d,
    alpha: f64,
) -> [Array2d; 4] {
    // Update the weights and biases by subtracting the gradients scaled by the learning rate (alpha).
    let w1 = w1 - alpha * dw1;
    let b1 = b1 - alpha * db1;
    let w2 = w2 - alpha * dw2;
    let b2 = b2 - alpha * db2;

    // Return the updated weights and biases.
    [w1, b1, w2, b2]
}

/// Gets the predicted labels by selecting the index of the highest output value
/// from the final layer's activations (a2). This assumes the network outputs
/// class probabilities and selects the most probable class.
///
/// # Parameters
/// - `a2`: The activations (output probabilities) from the final layer of the network.
///
/// # Returns
/// A vector of predicted labels (indices of the highest activation values).
fn get_predictions(a2: &Array2<f64>) -> Vec<usize> {
    // For each column (sample) in the activation matrix `a2`, find the index with the maximum value.
    a2.columns()
        .into_iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .filter_map(|(i, &value)| {
                    if value.is_nan() {
                        None // Skip NaN values.
                    } else {
                        Some((i, value))
                    }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less)) // Get the index of the max value.
                .map(|(i, _)| i) // Return the index of the maximum value (predicted class).
                .unwrap_or(0) // Default to 0 if no valid value is found.
        })
        .collect()
}

/// Calculates the accuracy of the model by comparing the predicted labels
/// to the true labels (y). The accuracy is the percentage of correct predictions
/// out of the total number of examples.
///
/// # Parameters
/// - `predictions`: The predicted labels (indices of the highest output values).
/// - `y`: The true labels.
///
/// # Returns
/// The accuracy of the model as a floating-point value between 0 and 1.
fn get_accuracy(predictions: Vec<usize>, y: &Array1d) -> f64 {
    // Count how many predictions match the true labels.
    let correct = zip(predictions, y)
        .filter(|(pred, label)| *pred as f64 == **label) // Check if predicted label equals true label.
        .count();

    // Return the accuracy as the ratio of correct predictions to total examples.
    correct as f64 / y.len_of(Axis(0)) as f64
}

/// Performs gradient descent for training the neural network.
/// Updates parameters to minimize cross-entropy loss.
///
/// # Arguments
/// - `x`: Input data matrix \( X \in \mathbb{R}^{784 \times m} \)
/// - `y`: Target labels \( y \in \mathbb{R}^{m} \)
/// - `iters`: Number of training iterations
/// - `alpha`: Learning rate \( \alpha \)
/// - `hidden_neurons`: Number of hidden layer neurons \( h \)
///
/// # Returns
/// Tuple of trained parameters: \( (W_1, b_1, W_2, b_2) \)
fn gradient_descent(
    x: Array2d,
    y: Array1d,
    iters: u32,
    alpha: f64,
    neurons: usize,
) -> [Array2d; 4] {
    // Initialize the parameters: weights and biases for the two layers.
    let (mut w1, mut b1, mut w2, mut b2) = init_params(neurons).into();

    // Gradient descent loop for a specified number of iterations.
    for i in 0..iters {
        // Perform forward propagation to get activations.
        let (z1, a1, _z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &x).into();

        // Perform backpropagation to get the gradients for weights and biases.
        let (dw1, db1, dw2, db2) = back_prop(&z1, &a1, &a2, &w2, &x, &y).into();

        // Update parameters using the gradients and learning rate.
        (w1, b1, w2, b2) = update_params(&w1, &b1, &w2, &b2, &dw1, &db1, &dw2, &db2, alpha).into();

        // Every 10 iterations, print out the current accuracy for monitoring.
        if i % 10 == 0 {
            println!("Iteration: {i}");
            println!(
                "Accuracy: {:.2}%",
                100. * get_accuracy(get_predictions(&a2), &y)
            );
        }
    }

    // Return the trained parameters (weights and biases).
    [w1, b1, w2, b2]
}

/// Makes predictions using the trained model by performing forward propagation
/// and then extracting the predicted labels from the final activations.
///
/// # Parameters
/// - `x`: The input data.
/// - `w1`: Weights for the first layer.
/// - `b1`: Biases for the first layer.
/// - `w2`: Weights for the second layer.
/// - `b2`: Biases for the second layer.
///
/// # Returns
/// A vector of predicted labels (indices of the highest activation values).
fn make_predictions(
    x: &Array2d,
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
) -> Vec<usize> {
    // Perform forward propagation to get the activations of the final layer.
    let (_, _, _, a2) = forward_prop(w1, b1, w2, b2, x).into();

    // Extract the predicted labels (indices of the highest activations).
    get_predictions(&a2)
}

/// Displays a prediction for a single test sample, showing the image and predicted label.
///
/// # Parameters
/// - `index`: The index of the sample to test.
/// - `x_train`: The training data (input samples).
/// - `y_train`: The true labels.
/// - `w1`: Weights for the first layer.
/// - `b1`: Biases for the first layer.
/// - `w2`: Weights for the second layer.
/// - `b2`: Biases for the second layer.
fn test_prediction(
    index: usize,
    x_train: &Array2<f64>,
    y_train: &Array1d,
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
) {
    // Get the input image at the specified index.
    let current_image = x_train.slice(s![.., index]).to_owned();

    // Perform prediction using the trained parameters.
    let prediction = make_predictions(&current_image.clone().insert_axis(Axis(1)), w1, b1, w2, b2);
    let label = y_train[index];

    // Output the predicted label and true label.
    println!("Prediction: {:?}", prediction[0]);
    println!("Label: {:?}", label);

    // Display the image along with the prediction and true label.
    show_image(&current_image, prediction[0], label, index);
}

/// Converts the image into a visual format and saves it to a file.
///
/// # Parameters
/// - `image`: The image data as a 1D array.
/// - `prediction`: The predicted label for the image.
/// - `label`: The true label for the image.
/// - `index`: The index of the image sample.
fn show_image(image: &Array1d, prediction: usize, label: f64, index: usize) {
    // Generate a file name based on prediction, label, and index.
    let file_name = format!("output-p{prediction}-l{label}-i{index}.png");
    let root = BitMapBackend::new(&file_name, (280, 280)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let pixel_size = 10;

    // Loop through the image pixels and draw each pixel as a rectangle on the canvas.
    for (i, val) in image.iter().enumerate() {
        let row = i / 28;
        let col = i % 28;
        let gray = (*val * 255.0) as u8;
        let color = RGBColor(gray, gray, gray);

        root.draw(&Rectangle::new(
            [
                (col as i32 * pixel_size, row as i32 * pixel_size),
                ((col + 1) as i32 * pixel_size, (row + 1) as i32 * pixel_size),
            ],
            color.filled(),
        ))
        .unwrap();
    }

    // Output the saved image file name.
    println!("Image saved to {file_name}");
}
