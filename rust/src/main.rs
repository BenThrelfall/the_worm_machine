use std::{fs::File, io::BufReader, os::unix::raw::time_t};

use neuron::Network;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    evolution::{evaluate, World},
    factory::Factory,
};

mod evolution;
mod factory;
mod neuron;

#[derive(Serialize, Deserialize)]
struct Frame {
    pub time: f64,
    pub data: Vec<f64>,
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let neurons: Vec<String>;
    let flat_g_syn: Vec<f64>;
    let flat_e_syn: Vec<f64>;
    let flat_g_gap: Vec<f64>;

    let mut time_trace: Vec<Frame>;

    let file = File::open("processed_data/time_trace.json").unwrap();
    let buffer = BufReader::new(file);
    time_trace = serde_json::from_reader(buffer).unwrap();

    for frame in time_trace.iter_mut() {
        for point in frame.data.iter_mut() {
            *point *= 10.0;
        }
    }

    let file = File::open("processed_data/default_g_syn.json").unwrap();
    let buffer = BufReader::new(file);
    flat_g_syn = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/default_e_syn.json").unwrap();
    let buffer = BufReader::new(file);
    flat_e_syn = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/default_g_gap.json").unwrap();
    let buffer = BufReader::new(file);
    flat_g_gap = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/neurons.json").unwrap();
    let buffer = BufReader::new(file);
    neurons = serde_json::from_reader(buffer).unwrap();

    let mut full_syn_g = Vec::new();
    let mut full_syn_e = Vec::new();
    let mut full_gap_g = Vec::new();

    for i in 0..neurons.len() {
        full_gap_g.push(Vec::new());
        full_syn_g.push(Vec::new());
        full_syn_e.push(Vec::new());

        for j in 0..neurons.len() {
            full_gap_g[i].push(flat_g_gap[i * neurons.len() + j]);
            full_syn_g[i].push(flat_g_syn[i * neurons.len() + j]);
            full_syn_e[i].push(flat_e_syn[i * neurons.len() + j]);
        }
    }

    let factory = Factory::new(full_syn_g, full_gap_g);
    let specification = factory.get_specification();

    let mut world = World::new();
    let mut population = world.random_population(&specification, 100);

    use std::time::Instant;
    let now = Instant::now();

    let heat = 1f64;

    for i in 0..3 {
        
        let results: Vec<f64> = population
            .par_iter_mut()
            .map(|genome| factory.build(genome.clone()))
            .map(|mut model| evaluate(&mut model, 20, &time_trace))
            .collect();

        let total: f64 = results.iter().sum();
        println!("Total Error {}", total);

        population = world.selection(&population, &results, 10);

        world.crossover(&mut population);
        world.mutate(&mut population, 0.25, 0.25, heat);
    }

    let elapsed = now.elapsed();

    println!("Elapsed: {:.2?}", elapsed);
}
