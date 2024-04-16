use std::{fs::File, io::{BufReader, BufWriter}};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    data::Frame, evolution::{evaluate, evaluate_std, predict, SmallGenome, SynapseType, World}, factory::Factory, neuron::{self, Network}
};

pub fn evolutionary_training(){

    let (time_trace, full_syn_g, full_gap_g, full_syn_e) = read_data();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
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

fn read_data() -> (Vec<Frame>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {

    let neurons: Vec<String>;
    let flat_g_syn: Vec<f64>;
    let flat_e_syn: Vec<f64>;
    let flat_g_gap: Vec<f64>;

    let mut time_trace: Vec<Frame>;

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

    (time_trace, full_syn_g, full_gap_g, full_syn_e)
}

fn preprocess_frames(time_trace: &mut Vec<Frame>, pre_adjust: f64, multiplier: f64, post_adjust: f64){
    for frame in time_trace.iter_mut() {
        for point in frame.data.iter_mut() {
            *point = multiplier * (*point + pre_adjust) + post_adjust;
        }
    }
}

pub fn experimental_run(){

    let (mut time_trace, full_syn_g, full_gap_g, full_syn_e) = read_data();

    let file = File::open("processed_data/sensory_indices.json").unwrap();
    let buffer = BufReader::new(file);
    let sensory_indices : Vec<usize> = serde_json::from_reader(buffer).unwrap();

    preprocess_frames(&mut time_trace, 0.0, 10.0, -35.0);

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let syn_types : Vec<SynapseType> = full_syn_e.iter().flatten().zip(full_syn_g.iter().flatten()).filter(|(e, g)| **g != 0.0).map(|(e, g)| if *e == 0.0{
        SynapseType::Excitatory
    }else{
        SynapseType::Inhibitory
    }).collect();

    println!("{}", specification.syn_len);
    println!("{}", syn_types.len());

    let genome = SmallGenome{
        syn_g: 100.0,
        syn_e_in: -45.0,
        syn_e_ex: 0.0,
        syn_types,
        gap_g: 100.0,
        gate_beta: 0.125,
        gate_adjust: -15.0,
        leak_g: 10.0,
        leak_e: -35.0,
    };

    let mut model = factory.build(genome.expand(&specification));

    let mut voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let mut gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let record;

    (_, _, record, _) = model.recorded_run_sensory(voltage, gates, 0.001, 10.0, &time_trace, &sensory_indices, 1);

    let file = File::create("processed_data/results.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &record).unwrap();

}