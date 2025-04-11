use ndarray::{Axis, s};
use polars::{io::SerReader, prelude::*};
use rand::seq::SliceRandom;

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

    println!("{m}, {n}");

    let data_train = data.slice(s![1000..m, ..]).t();
    // let x_dev = x_dev_slice / 255.;

    // let data_dev = data_dev_slice.t();
    // let y_dev = data_dev[0].clone();
}
