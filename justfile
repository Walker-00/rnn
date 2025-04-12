alias d := debug
alias r := release
alias nr := native_release

debug:
  cargo run

release:
  cargo run --release

native_release:
  RUSTFLAGS="-C target-cpu=native" cargo run --release
