use std::{fs::File, io::{BufReader, BufWriter}, os::unix::raw::time_t};

use neuron::Network;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    evolution::{evaluate, evaluate_std, predict, World},
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

    let mut heat = 1f64;

    let mut prev_total = 2097840300f64;
    let mut total = 2097840300f64;

    for i in 0..100 {
        
        let results: Vec<f64> = population
            .par_iter_mut()
            .map(|genome| factory.build(genome.clone()))
            .map(|mut model| evaluate(&mut model, 20, &time_trace))
            .collect();

        prev_total = total;
        total = results.iter().sum();

        let change = (prev_total - total) / prev_total;

        if change.abs() < 0.001{
            //heat += 0.1;
        }

        println!("Total Error {} Error Change {} Heat {}", total, change, heat);

        population = world.selection(&population, &results, 10);

        if i % 500 == 0{
            let file = File::create("processed_data/latest_population.json").unwrap();
            let buffer = BufWriter::new(file);
            serde_json::to_writer(buffer, &population).unwrap();
        }

        world.crossover(&mut population);
        world.mutate(&mut population, 0.25, 0.25, heat);

        heat *= 0.99;
    }

    let results: Vec<f64> = population
        .par_iter_mut()
        .map(|genome| factory.build(genome.clone()))
        .map(|mut model| evaluate(&mut model, 20, &time_trace))
        .collect();

    population = world.selection(&population, &results, 10);

    let file = File::create("processed_data/final_population.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &population).unwrap();

    let best = population.first().unwrap();
    let mut best_model = factory.build(best.clone());

    let final_data = predict(&mut best_model, 20, &time_trace);

    let file = File::create("processed_data/prediction.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &final_data).unwrap();

    let elapsed = now.elapsed();

    println!("Elapsed: {:.2?}", elapsed);
}
