use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr, s};
use polars::{io::SerReader, prelude::*};
use rand::{Rng, distr::Uniform, seq::SliceRandom};

type NDArray = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

// type Wnb = [(NDArray, NDArray); 2];

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

    let y_train = data_train.slice(s![0, ..]);
    let x_train_slice = data_train.slice(s![1..n, ..]);
    let x_train = x_train_slice.to_owned() / 255.0;

    println!("{y_train}");
    println!("{:?}", x_train.slice(s![.., 0]).dim());
}

fn init_params() -> [NDArray; 4] {
    let mut rng = rand::rng();

    let dist = Uniform::new(0., 1.).unwrap();

    let w1 = Array2::from_shape_fn((10, 784), |_| rng.sample(dist)) - 0.5;
    let b1 = Array2::from_shape_fn((10, 1), |_| rng.sample(dist)) - 0.5;

    let w2 = Array2::from_shape_fn((10, 10), |_| rng.sample(dist)) - 0.5;
    let b2 = Array2::from_shape_fn((10, 1), |_| rng.sample(dist)) - 0.5;

    [w1, b1, w2, b2]
}

fn forward_prop(w1: NDArray, b1: NDArray, w2: NDArray, b2: NDArray, x: NDArray) {
    let z1 = w1.dot(&x) + b1;
}
