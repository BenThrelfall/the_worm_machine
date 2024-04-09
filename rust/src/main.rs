use std::{fs::File, io::BufReader};

use neuron::Network;

mod neuron;

fn main() {

    let neurons: Vec<String>;
    let flat_g_syn: Vec<f64>;
    let flat_e_syn: Vec<f64>;
    let flat_g_gap: Vec<f64>;

    let file =  File::open("processed_data/default_g_syn.json").unwrap();
    let buffer = BufReader::new(file);
    flat_g_syn = serde_json::from_reader(buffer).unwrap();

    let file =  File::open("processed_data/default_e_syn.json").unwrap();
    let buffer = BufReader::new(file);
    flat_e_syn = serde_json::from_reader(buffer).unwrap();

    let file =  File::open("processed_data/default_g_gap.json").unwrap();
    let buffer = BufReader::new(file);
    flat_g_gap = serde_json::from_reader(buffer).unwrap();

    let file =  File::open("processed_data/neurons.json").unwrap();
    let buffer = BufReader::new(file);
    neurons= serde_json::from_reader(buffer).unwrap();

    let mut full_syn_g = Vec::new();
    let mut full_syn_e = Vec::new();
    let mut full_gap_g = Vec::new();

    for i in 0..neurons.len(){

        full_gap_g.push(Vec::new());
        full_syn_g.push(Vec::new());
        full_syn_e.push(Vec::new());

        for j in 0..neurons.len(){
            full_gap_g[i].push(flat_g_gap[i * neurons.len() + j]);
            full_syn_g[i].push(flat_g_syn[i * neurons.len() + j]);
            full_syn_e[i].push(flat_e_syn[i * neurons.len() + j]);
        }
    }

    let gate_beta = vec![0.125f64; 280];
    let gate_adjust = vec![-15f64; 280];
    let leak_g = vec![10f64; 280];
    let leak_e = vec![-35f64; 280];

    let mut model = Network::new(full_syn_g, full_syn_e, gate_beta, gate_adjust, leak_g, leak_e, full_gap_g);

    let voltage = vec![-60f64; 280];
    let gates = vec![0f64; 280];

    use std::time::Instant;
    let now = Instant::now();

    let results = model.run(voltage, gates, 0.001, 1.0);

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    
    println!("{:?}", results.0);
}
