
mod evolution;
mod factory;
mod neuron;
mod programs;
mod data;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    //programs::evolutionary_training();
    programs::experimental_run();
}
