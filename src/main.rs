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
    let data_dev_slice = data.slice(s![0..1000, ..]);
    let data_dev = data_dev_slice.t(); // shape: (n, 1000) => transpose for column-major

    let y_dev = data_dev.slice(s![0, ..]);
    let x_dev_slice = data_dev.slice(s![1..n, ..]);
    let x_dev = x_dev_slice.to_owned() / 255.0;

    let data_train_slice = data.slice(s![1000..m, ..]);
    let data_train = data_train_slice.t();

    let y_train = data_train.row(0);
    let x_train_slice = data_train.slice(s![1..n, ..]);
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
    let w1 = Array2::from_shape_fn((neurons, 784), |_| rng.sample(dist)) - 0.5;
    let b1 = Array2::from_shape_fn((neurons, 1), |_| rng.sample(dist)) - 0.5;

    let w2 = Array2::from_shape_fn((10, neurons), |_| rng.sample(dist)) - 0.5;
    let b2 = Array2::from_shape_fn((10, 1), |_| rng.sample(dist)) - 0.5;

    [w1, b1, w2, b2]
}

/// Applies the ReLU activation function element-wise.
///
/// \[ \text{ReLU}(x) = \max(0, x) \]
fn relu(z: &Array2d) -> Array2d {
    let mut z = z.clone();
    z.par_mapv_inplace(|v| v.max(0.0));
    z
}

/// Computes the derivative of ReLU for backpropagation.
///
/// \[ \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases} \]
fn deriv_relu(z: &Array2d) -> Array2d {
    z.mapv(|v| if v > 0. { 1. } else { 0. })
}

/// Applies the softmax function across the output layer.
/// Ensures output probabilities sum to 1 for each column (sample).
///
/// \[ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \]
fn softmax(z: &Array2d) -> Array2d {
    let max_per_col = z.map_axis(Axis(0), |col| {
        col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });
    let shifted = z - &max_per_col.insert_axis(Axis(0));
    let exp = shifted.mapv(|x| x.exp());
    let sum_exp = exp.sum_axis(Axis(0)).insert_axis(Axis(0));
    &exp / &sum_exp
}

fn forward_prop(
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
    x: &Array2d,
) -> [Array2d; 4] {
    let z1 = w1.dot(x) + b1;
    let a1 = relu(&z1);
    let z2 = w2.dot(&a1) + b2;
    let a2 = softmax(&z2);

    [z1, a1, z2, a2]
}

/// Converts labels into one-hot encoded matrix form.
/// Used for computing gradients and loss.
fn one_hot(y: &Array1d) -> Array2d {
    let max = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let num_samples = y.len();
    let num_classes = *max as usize + 1;
    let mut one_hot_y = Array2::<f64>::zeros((num_samples, num_classes));

    for (i, class_idx) in y.iter().enumerate() {
        one_hot_y[[i, *class_idx as usize]] = 1.;
    }

    one_hot_y.t().to_owned()
}

fn back_prop(
    z1: &Array2d,
    a1: &Array2d,
    // z2: Array2d,
    a2: &Array2d,
    w2: &Array2d,
    x: &Array2d,
    y: &Array1d,
) -> [Array2d; 4] {
    let m = y.len();
    let one_hot_y = one_hot(y);

    let dz2 = a2 - one_hot_y;
    let dw2 = 1. / m as f64 * dz2.dot(&a1.t());
    // let db2 = 1. / m as f64 * dz2.sum_axis(Axis(2));
    let db2 = dz2.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1. / m as f64);

    let dz1 = w2.t().dot(&dz2) * deriv_relu(z1);
    let dw1 = 1. / m as f64 * dz1.dot(&x.t());
    // let db1 = 1. / m as f64 * dz1.sum_axis(Axis(2));
    let db1 = dz1.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1. / m as f64);

    [dw1, db1, dw2, db2]
}

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
    let w1 = w1 - alpha * dw1;
    let b1 = b1 - alpha * db1;

    let w2 = w2 - alpha * dw2;
    let b2 = b2 - alpha * db2;

    [w1, b1, w2, b2]
}

fn get_predictions(a2: &Array2<f64>) -> Vec<usize> {
    a2.columns()
        .into_iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .filter_map(|(i, &value)| {
                    if value.is_nan() {
                        None
                    } else {
                        Some((i, value))
                    }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

fn get_accuracy(predictions: Vec<usize>, y: &Array1d) -> f64 {
    // println!("{predictions:?} {y}");

    let correct = zip(predictions, y)
        .filter(|(pred, label)| *pred as f64 == **label)
        .count();

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
    let (mut w1, mut b1, mut w2, mut b2) = init_params(neurons).into();

    for i in 0..iters {
        let (z1, a1, _z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &x).into();
        let (dw1, db1, dw2, db2) = back_prop(&z1, &a1, &a2, &w2, &x, &y).into();
        (w1, b1, w2, b2) = update_params(&w1, &b1, &w2, &b2, &dw1, &db1, &dw2, &db2, alpha).into();

        if i % 10 == 0 {
            println!("Iteration: {i}");
            println!(
                "Accuracy: {:.2}%",
                100. * get_accuracy(get_predictions(&a2), &y)
            );
        }
    }

    [w1, b1, w2, b2]
}

fn make_predictions(
    x: &Array2d,
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
) -> Vec<usize> {
    let (_, _, _, a2) = forward_prop(w1, b1, w2, b2, x).into();
    get_predictions(&a2)
}

/// Displays a prediction for a single test sample, showing the image and predicted label.
fn test_prediction(
    index: usize,
    x_train: &Array2<f64>,
    y_train: &Array1d,
    w1: &Array2d,
    b1: &Array2d,
    w2: &Array2d,
    b2: &Array2d,
) {
    // Get the column at the specified index
    let current_image = x_train.slice(s![.., index]).to_owned();

    // Run prediction
    let prediction = make_predictions(&current_image.clone().insert_axis(Axis(1)), w1, b1, w2, b2);
    let label = y_train[index];

    println!("Prediction: {:?}", prediction[0]);
    println!("Label: {:?}", label);

    // Convert to image
    show_image(&current_image, prediction[0], label, index);
}

fn show_image(image: &Array1d, prediction: usize, label: f64, index: usize) {
    let file_name = format!("output-p{prediction}-l{label}-i{index}.png");
    let root = BitMapBackend::new(&file_name, (280, 280)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let pixel_size = 10;
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

    println!("Image saved to {file_name}");
}

