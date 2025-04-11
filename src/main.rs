use ndarray::Axis;
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
    let data_shapes = data.shape();
    let (m, n) = (data_shapes[0], data_shapes[1]);
    let mut rows: Vec<_> = data.axis_iter(Axis(0)).collect();
    rows.shuffle(&mut rng);

    let data = ndarray::stack(
        Axis(0),
        &rows.iter().map(|row| row.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    // let ndarray_data = ndarray::array![array_data];
    // let shape = ndarray_data.shape();
    //
    // let data_dev_slice = ndarray_data.slice(s![0..=1000]);
    // let data_dev = data_dev_slice.t();
    // let y_dev = data_dev[0].clone();
}
