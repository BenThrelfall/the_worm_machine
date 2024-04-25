use std::{fs::File, io::BufReader};

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Frame {
    pub time: f64,
    pub data: Vec<f64>,
}

pub fn preprocess_frames(time_trace: &mut Vec<Frame>, multiplier: f64, post_adjust: f64){
    for frame in time_trace.iter_mut() {
        for point in frame.data.iter_mut() {
            *point = multiplier * *point + post_adjust;
        }
    }
}

pub fn unprocess_frames(time_trace: &mut Vec<Frame>, multiplier: f64, post_adjust: f64){
    for frame in time_trace.iter_mut() {
        for point in frame.data.iter_mut() {
            *point = (*point - post_adjust) / multiplier;
        }
    }
}

pub fn read_data() -> (Vec<Frame>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {

    let neurons: Vec<String>;
    let flat_g_syn: Vec<f64>;
    let flat_e_syn: Vec<f64>;
    let flat_g_gap: Vec<f64>;

    let time_trace: Vec<Frame>;

    let file = File::open("processed_data/time_trace.json").unwrap();
    let buffer = BufReader::new(file);
    time_trace = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/new_g_syn.json").unwrap();
    let buffer = BufReader::new(file);
    flat_g_syn = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/default_e_syn.json").unwrap();
    let buffer = BufReader::new(file);
    flat_e_syn = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/new_g_gap.json").unwrap();
    let buffer = BufReader::new(file);
    flat_g_gap = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/neurons.json").unwrap();
    let buffer = BufReader::new(file);
    neurons = serde_json::from_reader(buffer).unwrap();

    let file = File::open("processed_data/sensory_indices.json").unwrap();
    let buffer = BufReader::new(file);
    let sensory_indices : Vec<usize> = serde_json::from_reader(buffer).unwrap();

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

    (time_trace, full_syn_g, full_gap_g, full_syn_e, sensory_indices)
}