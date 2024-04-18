#![allow(dead_code)]

mod evolution;
mod factory;
mod neuron;
mod programs;
mod data;
mod genetics;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    //programs::evolutionary_training();
    //programs::experimental_run();
    programs::gate_calculation();
}
