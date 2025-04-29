#![allow(dead_code)]

mod evolution;
mod factory;
mod neuron;
mod programs;
mod data;
mod genetics;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    //programs::evolutionary_training_no_calc_gates();
    //programs::evolutionary_training();
    //programs::plm_test_run();    
    programs::gate_calculation();
    //programs::small_evolutionary_training();
    //programs::two_node_patch_clamping();
}
