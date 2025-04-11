use std::{f64, iter::zip, usize};

use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr, s};
use polars::{frame::row, io::SerReader, prelude::*};
use rand::{Rng, distr::Uniform, seq::SliceRandom};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};

type Array2d = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

// type Wnb = [(Array2d, Array2d); 2];

// fn split_and_normalize(data: &Array2<f64>, from: usize, to: usize) -> (ArrayView1<f64>, Array2<f64>) {
//     let sliced = data.slice(s![from..to, ..]).t();
//     let y = sliced.slice(s![0, ..]);
//     let x = sliced.slice(s![1.., ..]).to_owned() / 255.0;
//     (y, x)
// }

fn main() {
    let mut rng = rand::rng();
    let csv_data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/train.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let data = csv_data.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    // data.slice();
    let (m, n) = data.dim();
    let mut rows: Vec<_> = data.axis_iter(Axis(0)).collect();
    rows.shuffle(&mut rng);

    let data = ndarray::stack(
        Axis(0),
        &rows.iter().map(|row| row.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let data_dev_slice = data.slice(s![0..1000, ..]);
    let data_dev = data_dev_slice.t();

    let y_dev = data_dev.slice(s![0, ..]);
    let x_dev_slice = data_dev.slice(s![1..n, ..]);
    let x_dev = x_dev_slice.to_owned() / 255.0;

    let data_train_slice = data.slice(s![1000..m, ..]);
    let data_train = data_train_slice.t();

    let y_train = data_train.row(0);
    let x_train_slice = data_train.slice(s![1..n, ..]);
    let x_train = x_train_slice.to_owned() / 255.0;

    println!("{y_train}");
    println!("{:?}", x_train.slice(s![.., 0]).dim());

    let (w1, b1, w2, b2) = gradient_descent(x_train, y_train, 100, 0.1).into();
}

fn init_params() -> [Array2d; 4] {
    let mut rng = rand::rng();

    let dist = Uniform::new(0., 1.).unwrap();

    let w1 = Array2::from_shape_fn((10, 784), |_| rng.sample(dist)) - 0.5;
    let b1 = Array2::from_shape_fn((10, 1), |_| rng.sample(dist)) - 0.5;

    let w2 = Array2::from_shape_fn((10, 10), |_| rng.sample(dist)) - 0.5;
    let b2 = Array2::from_shape_fn((10, 1), |_| rng.sample(dist)) - 0.5;

    [w1, b1, w2, b2]
}

fn relu(z: &Array2d) -> Array2d {
    let mut z = z.clone();
    z.par_mapv_inplace(|v| v.max(0.0));
    z
}

fn deriv_relvu(z: &Array2d) -> Array2d {
    z.mapv(|v| if v > 0. { 1. } else { 0. })
}

fn softmax(z: &Array2d) -> Array2d {
    z.exp() / z.exp().sum()
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

fn one_hot(y: &Array2d) -> Array2d {
    let max = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let num_samples = y.len();
    let num_classes = *max as usize + 1;
    let mut one_hot_y = Array2::<f64>::zeros((num_samples, num_classes));

    // let mut one_hot_y = Array2::<f64>::zeros(((y.len() as f64), max + 1.));

    // let rows = one_hot_y.iter().cloned().enumerate().collect::<Vec<_>>();

    // rows.into_par_iter().for_each(|(i, class_idx)| {
    //     one_hot_y[[i, class_idx as usize]] = 1.;
    // });

    // rows.par_iter_mut().enumerate().for_each(|(i, row)| {
    //     let class_idx = y[i as f64];
    //     row[class_idx] = 1.
    // });

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
    y: &Array2d,
) -> [Array2d; 4] {
    let m = y.len();
    let one_hot_y = one_hot(y);

    let dz2 = a2 - one_hot_y;
    let dw2 = 1. / m as f64 * dz2.dot(&a1.t());
    // let db2 = 1. / m as f64 * dz2.sum_axis(Axis(2));
    let db2 = dz2.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1. / m as f64);

    let dz1 = w2.t().dot(&dz2) * deriv_relvu(z1);
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

fn get_predictions(a2: &Array2d) -> Vec<usize> {
    a2.columns()
        .into_iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect()
}

fn get_accuracy(predictions: Vec<usize>, y: &Array2d) -> f64 {
    println!("{predictions:?} {y}");

    // let label_classes: Vec<usize> = y
    //     .axis_iter(Axis(0))
    //     .map(|row| {
    //         row.iter()
    //             .enumerate()
    //             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    //             .map(|(i, _)| i)
    //             .unwrap()
    //     })
    //     .collect();

    let correct = zip(predictions, y)
        .filter(|(pred, label)| *pred as f64 == **label)
        .count();

    correct as f64 / y.len_of(Axis(0)) as f64
}

fn gradient_descent(x: Array2d, y: Array2d, iters: u32, alpha: f64) -> [Array2d; 4] {
    let (mut w1, mut b1, mut w2, mut b2) = init_params().into();

    for i in 0..iters {
        let (z1, a1, z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &x).into();
        let (dw1, db1, dw2, db2) = back_prop(&z1, &a1, &a2, &w2, &x, &y).into();
        (w1, b1, w2, b2) = update_params(&w1, &b1, &w2, &b2, &dw1, &db1, &dw2, &db2, alpha).into();

        if i % 10 == 0 {
            println!("Iteration: {i}");
            println!("Accuracy: {}", get_accuracy(get_predictions(&a2), &y));
        }
    }

    [w1, b1, w2, b2]
}
