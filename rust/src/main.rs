#![allow(dead_code)]

mod evolution;
mod factory;
mod neuron;
mod programs;
mod data;
mod genetics;


fn main() {
    //std::env::set_var("RUST_BACKTRACE", "1");

    programs::evolutionary_training();
    programs::evolutionary_training_no_calc_gates();
    //programs::gate_calculation();
    //programs::small_evolutionary_training();
}
