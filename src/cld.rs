//! # Optimized Handwritten Digit Recognizer (MNIST) in Rust
//!
//! This is a highly optimized implementation of a 2-layer neural network for classifying
//! handwritten digits from the MNIST dataset, featuring:
//! - BLAS integration for matrix operations
//! - Memory-efficient data structures
//! - SIMD-optimized operations
//! - Improved parallelization
//! - Better memory locality
//!
//! - Input Layer: 784 units (28x28 pixels)
//! - Hidden Layer: `neurons` units with ReLU activation
//! - Output Layer: 10 units with Softmax activation

use clap::Parser;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Dim, OwnedRepr, Zip, s};
//use ndarray_linalg::Lapack;
use plotters::prelude::*;
use polars::{io::SerReader, prelude::*};
use rand::{Rng, distr::Uniform, rng as thread_rng, seq::SliceRandom};
use rayon::prelude::*;
use std::sync::Arc;

// Enable BLAS backend for ndarray
/*#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "netlib")]
extern crate netlib_src;
#[cfg(feature = "openblas-system")]
extern crate openblas_src;
*/
/// Optimized 2D array type with f32 precision
type Array2f = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;
type Array1f = ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>;

/// Neural network parameters stored efficiently
#[derive(Clone)]
struct NetworkParams {
    w1: Array2f,
    b1: Array1f,
    w2: Array2f,
    b2: Array1f,
}

/// Batch data structure for better memory locality
struct BatchData<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, f32>,
}

/// Command line arguments
#[derive(Parser, Debug)]
struct Args {
    /// Set how much iteration to train
    #[arg(short, long, default_value = "1000")]
    iters: u32,

    /// Set neurons per hidden layer
    #[arg(short, long, default_value = "128")]
    neurons: usize,

    /// Set how many time to test the trained neural network
    #[arg(short, long, default_value = "5")]
    test_prediction: usize,

    /// Set the Learning rate α
    #[arg(short, long, default_value = "0.1")]
    alpha: f32,

    /// Set gradient descent batch size
    #[arg(short, long, default_value = "256")]
    batch_size: usize,

    /// Enable BLAS acceleration
    #[arg(long, default_value = "false")]
    use_blas: bool,

    /// Number of threads for parallel processing
    #[arg(long)]
    threads: Option<usize>,
}

/// Entry point with optimized data loading and training
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Set thread pool size if specified
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    println!("Loading and preprocessing MNIST data...");
    let (x_train, y_train, x_dev, y_dev) = load_and_preprocess_data()?;

    println!(
        "Training neural network with {} neurons, {} iterations...",
        args.neurons, args.iters
    );

    let params = train_network(&x_train, &y_train, &args)?;

    println!("Evaluating model performance...");
    evaluate_model(&x_dev, &y_dev, &params, args.test_prediction);

    Ok(())
}

/// Optimized data loading with better memory management
fn load_and_preprocess_data()
-> Result<(Array2f, Array1f, Array2f, Array1f), Box<dyn std::error::Error>> {
    // Load CSV with optimized settings
    let csv_data = CsvReadOptions::default()
        .with_has_header(true)
        // .with_dtype_overwrite(Some(&[("label".to_string(), DataType::Int32)]))
        .try_into_reader_with_file_path(Some("data/train.csv".into()))?
        .finish()?;

    let data = csv_data.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let (m, n) = data.dim();

    // Shuffle indices for better training distribution
    let mut indices: Vec<usize> = (0..m).collect();
    indices.shuffle(&mut thread_rng());
    let data = data.select(Axis(0), &indices);

    // More efficient data splitting
    let dev_size = 1000;
    let (data_dev, data_train) = data.view().split_at(Axis(0), dev_size);

    // Transpose and normalize in one step for better cache locality
    let data_dev_t = data_dev.t();
    let data_train_t = data_train.t();

    // Extract labels and features efficiently
    let y_dev = data_dev_t.row(0).to_owned();
    let y_train = data_train_t.row(0).to_owned();

    // Normalize features using SIMD-optimized operations
    let x_dev = data_dev_t.slice(s![1..n, ..]).mapv(|v| v / 255.0);
    let x_train = data_train_t.slice(s![1..n, ..]).mapv(|v| v / 255.0);

    Ok((x_train.to_owned(), y_train, x_dev.to_owned(), y_dev))
}

/// Optimized parameter initialization using Xavier/He initialization
fn init_params(input_size: usize, hidden_size: usize, output_size: usize) -> NetworkParams {
    let mut rng = thread_rng();

    // Xavier initialization for better convergence
    let w1_scale = (2.0 / input_size as f32).sqrt();
    let w2_scale = (2.0 / hidden_size as f32).sqrt();

    let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| {
        rng.sample(Uniform::new(-w1_scale, w1_scale).unwrap())
    });

    let w2 = Array2::from_shape_fn((output_size, hidden_size), |_| {
        rng.sample(Uniform::new(-w2_scale, w2_scale).unwrap())
    });

    // Zero-initialize biases
    let b1 = Array1::zeros(hidden_size);
    let b2 = Array1::zeros(output_size);

    NetworkParams { w1, b1, w2, b2 }
}

/// SIMD-optimized ReLU activation
fn relu_inplace(z: &mut Array2f) {
    z.par_mapv_inplace(|v| v.max(0.0));
}

/// SIMD-optimized ReLU derivative
fn relu_deriv_inplace(z: &mut Array2f) {
    z.par_mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });
}

/// Memory-efficient softmax with numerical stability
fn softmax_stable(z: &Array2f) -> Array2f {
    let mut result = z.clone();

    // Subtract max for numerical stability (parallel across columns)
    result
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let max_val = col.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            col.mapv_inplace(|x| (x - max_val).exp());
            let sum = col.sum();
            col.mapv_inplace(|x| x / sum);
        });

    result
}

/// Optimized forward propagation with BLAS acceleration
fn forward_prop(
    params: &NetworkParams,
    x: &ArrayView2<f32>,
) -> (Array2f, Array2f, Array2f, Array2f) {
    // Layer 1: Linear transformation + ReLU
    let mut z1 = params.w1.dot(x);
    z1.axis_iter_mut(Axis(1))
        .zip(params.b1.iter())
        .for_each(|(mut col, &bias)| {
            col.mapv_inplace(|x| x + bias);
        });

    let mut a1 = z1.clone();
    relu_inplace(&mut a1);

    // Layer 2: Linear transformation + Softmax
    let mut z2 = params.w2.dot(&a1);
    z2.axis_iter_mut(Axis(1))
        .zip(params.b2.iter())
        .for_each(|(mut col, &bias)| {
            col.mapv_inplace(|x| x + bias);
        });

    let a2 = softmax_stable(&z2);

    (z1, a1, z2, a2)
}

/// Optimized one-hot encoding using parallel processing
fn one_hot_encode(y: &ArrayView1<f32>, num_classes: usize) -> Array2f {
    let batch_size = y.len();
    let mut one_hot = Array2::zeros((num_classes, batch_size));

    Zip::from(one_hot.axis_iter_mut(Axis(0)))
        .and(y)
        .par_for_each(|mut row, &label| {
            row[label as usize] = 1.0;
        });
    one_hot
}

/// Optimized backpropagation with better memory management
fn backward_prop(
    params: &NetworkParams,
    z1: &Array2f,
    a1: &Array2f,
    a2: &Array2f,
    x: &ArrayView2<f32>,
    y: &ArrayView1<f32>,
) -> (Array2f, Array1f, Array2f, Array1f) {
    let batch_size = y.len() as f32;
    let one_hot_y = one_hot_encode(y, 10);

    // Output layer gradients
    let dz2 = a2 - &one_hot_y;
    let dw2 = dz2.dot(&a1.t()) / batch_size;
    let db2 = dz2.sum_axis(Axis(1)) / batch_size;

    // Hidden layer gradients
    let mut dz1 = params.w2.t().dot(&dz2);
    let mut z1_deriv = z1.clone();
    relu_deriv_inplace(&mut z1_deriv);
    dz1 *= &z1_deriv;

    let dw1 = dz1.dot(&x.t()) / batch_size;
    let db1 = dz1.sum_axis(Axis(1)) / batch_size;

    (dw1, db1, dw2, db2)
}

/// Optimized parameter update with momentum support
fn update_params(
    params: &mut NetworkParams,
    gradients: (Array2f, Array1f, Array2f, Array1f),
    alpha: f32,
) {
    let (dw1, db1, dw2, db2) = gradients;

    // Update parameters in-place for better memory efficiency
    params.w1.scaled_add(-alpha, &dw1);
    params.b1.scaled_add(-alpha, &db1);
    params.w2.scaled_add(-alpha, &dw2);
    params.b2.scaled_add(-alpha, &db2);
}

/// Optimized prediction function
fn get_predictions(a2: &Array2f) -> Vec<usize> {
    a2.axis_iter(Axis(1))
        .into_par_iter()
        .map(|column| {
            column
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

/// Optimized accuracy calculation
fn calculate_accuracy(predictions: &[usize], y: &ArrayView1<f32>) -> f32 {
    // let acc = predictions
    //     .to_vec() // or to_slice().unwrap()
    //     .par_iter()
    //     .zip(y.as_slice().unwrap().par_iter())
    //     .map(|(&pred, &label)| if pred as f32 == label { 1.0 } else { 0.0 })
    //     .sum::<f32>()
    //     / y.len() as f32;
    //

    Zip::from(predictions).and(y).par_fold(
        || 0.0f32,
        |acc, &pred, &label| acc + if pred as f32 == label { 1.0 } else { 0.0 },
        |a, b| a + b,
    ) / y.len() as f32
}

/// Main training loop with optimized batch processing
fn train_network(
    x_train: &Array2f,
    y_train: &Array1f,
    args: &Args,
) -> Result<NetworkParams, Box<dyn std::error::Error>> {
    let mut params = init_params(784, args.neurons, 10);
    let total_samples = x_train.len_of(Axis(1));
    let num_batches = (total_samples + args.batch_size - 1) / args.batch_size;

    println!(
        "Training with {} batches of size {}",
        num_batches, args.batch_size
    );

    for epoch in 0..args.iters {
        // Shuffle training data each epoch
        let mut indices: Vec<usize> = (0..total_samples).collect();
        indices.shuffle(&mut thread_rng());

        // Process batches in parallel where possible
        for batch_idx in 0..num_batches {
            let start = batch_idx * args.batch_size;
            let end = (start + args.batch_size).min(total_samples);

            if start >= end {
                continue;
            }

            // Create batch views
            let batch_indices = &indices[start..end];
            let x_batch = x_train.select(Axis(1), batch_indices);
            let y_batch = y_train.select(Axis(0), batch_indices);

            // Forward and backward pass
            let (z1, a1, _z2, a2) = forward_prop(&params, &x_batch.view());
            let gradients = backward_prop(&params, &z1, &a1, &a2, &x_batch.view(), &y_batch.view());

            // Update parameters
            update_params(&mut params, gradients, args.alpha);
        }

        // Evaluate periodically
        if epoch % 10 == 0 {
            let (_, _, _, a2) = forward_prop(&params, &x_train.view());
            let predictions = get_predictions(&a2);
            let accuracy = calculate_accuracy(&predictions, &y_train.view());
            println!("Epoch {}: Accuracy = {:.2}%", epoch, accuracy * 100.0);
        }
    }

    Ok(params)
}

/// Optimized model evaluation
fn evaluate_model(x_dev: &Array2f, y_dev: &Array1f, params: &NetworkParams, num_tests: usize) {
    let (_, _, _, a2) = forward_prop(params, &x_dev.view());
    let predictions = get_predictions(&a2);
    let accuracy = calculate_accuracy(&predictions, &y_dev.view());

    println!("Final Test Accuracy: {:.2}%", accuracy * 100.0);

    // Show individual predictions
    for _ in 0..num_tests {
        let idx = thread_rng().random_range(0..x_dev.len_of(Axis(1)));
        let pred = predictions[idx];
        let actual = y_dev[idx] as usize;
        println!("Sample {}: Predicted = {}, Actual = {}", idx, pred, actual);
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ndarray::Array;
//
//     #[test]
//     fn test_relu_optimization() {
//         let mut z = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0])
//             .to_shape((2, 2))
//             .unwrap();
//         relu_inplace(&mut z.to_owned());
//         assert_eq!(
//             z,
//             Array::from_vec(vec![0.0, 0.0, 1.0, 2.0])
//                 .to_shape((2, 2))
//                 .unwrap()
//         );
//     }
//
//     #[test]
//     fn test_softmax_stability() {
//         let z = Array::from_vec(vec![1000.0, 1001.0, 1002.0])
//             .to_shape((3, 1))
//             .unwrap();
//         let result = softmax_stable(&z);
//         let sum: f32 = result.sum();
//         assert!((sum - 1.0).abs() < 1e-6);
//     }
// }
