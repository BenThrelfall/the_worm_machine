use std::{fs::File, io::BufReader, os::unix::raw::time_t};

use neuron::Network;
use serde::{Deserialize, Serialize};

use crate::{evolution::{evaluate, World}, factory::Factory};

mod evolution;
mod factory;
mod neuron;

#[derive(Serialize, Deserialize)]
struct Frame {
    pub time: f64,
    pub data: Vec<f64>,
}

fn main() {
    let neurons: Vec<String>;
    let flat_g_syn: Vec<f64>;
    let flat_e_syn: Vec<f64>;
    let flat_g_gap: Vec<f64>;

    let time_trace: Vec<Frame>;

    let file = File::open("processed_data/time_trace.json").unwrap();
    let buffer = BufReader::new(file);
    time_trace = serde_json::from_reader(buffer).unwrap();

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

    use std::time::Instant;
    let now = Instant::now();

    let mut world = World::new();
    let population = world.random_population(&specification, 50);
    let results: Vec<f64> = population.iter()
                            .map(|genome| factory.build(genome.clone()))
                            .map(|mut model| evaluate(&mut model, 20, &time_trace))
                            .collect();

    let elapsed = now.elapsed();

    println!("Elapsed: {:.2?}", elapsed);
    println!("{:?}", results);
}
