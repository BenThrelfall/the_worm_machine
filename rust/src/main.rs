#![allow(dead_code)]

mod evolution;
mod factory;
mod neuron;
mod programs;
mod data;
mod genetics;


fn main() {
    //std::env::set_var("RUST_BACKTRACE", "1");

    //programs::evolutionary_training();
    //programs::experimental_run();
    //programs::gate_calculation();
    programs::small_evolutionary_training();
    //programs::proprocess_experiment_with_gate_calc();
    //programs::model_vs_null_performance();
}
