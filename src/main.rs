use csv::Reader;

fn main() {
    let mut data = Reader::from_path("data/train.csv").unwrap();
    println!("{:#?}", data.headers());
}
