use crate::{evolution::Genome, neuron::Network};

pub struct Factory {
    pub model_len: usize,
    pub syn_indices: Vec<Vec<usize>>,
    pub gap_indices: Vec<Vec<usize>>,
}

pub struct Specification {
    pub model_len: usize,
    pub syn_len: usize,
    pub gap_len: usize,
}

impl Factory {
    pub fn new(full_syn_g: Vec<Vec<f64>>, full_gap_g: Vec<Vec<f64>>) -> Self {
        let model_len = full_gap_g.len();

        let mut syn_indices = Vec::new();

        for (i, line) in full_syn_g.iter().enumerate() {
            syn_indices.push(Vec::new());

            for (j, val) in line.iter().enumerate() {
                if *val != 0f64 {
                    syn_indices[i].push(j);
                }
            }
        }

        let mut gap_indices = Vec::new();

        for (i, line) in full_gap_g.iter().enumerate() {
            gap_indices.push(Vec::new());

            for (j, val) in line.iter().enumerate() {
                if *val != 0f64 {
                    gap_indices[i].push(j);
                }
            }
        }

        Factory {
            model_len,
            syn_indices,
            gap_indices,
        }
    }

    pub fn get_specification(&self) -> Specification {
        let model_len = self.model_len;
        let syn_len = self.syn_indices.iter().map(|x| x.len()).sum();
        let gap_len = self.gap_indices.iter().map(|x| x.len()).sum();

        Specification {
            model_len,
            syn_len,
            gap_len,
        }
    }

    pub fn build(
        &self,
        genome : Genome,
    ) -> Network {

        let syn_indices = self.syn_indices.clone();
        let gap_indices = self.gap_indices.clone();

        let mut syn_g = Vec::new();
        let mut syn_e = Vec::new();

        let mut count = 0;
        for (i, line) in syn_indices.iter().enumerate(){

            syn_g.push(Vec::new());
            syn_e.push(Vec::new());

            for _ in line.iter(){
                syn_g[i].push(genome.flat_syn_g[count]);
                syn_e[i].push(genome.flat_syn_e[count]);
                count += 1;
            }
        }

        let mut gap_g = Vec::new();

        let mut count = 0;
        for (i, line) in gap_indices.iter().enumerate(){

            gap_g.push(Vec::new());

            for _ in line.iter(){
                gap_g[i].push(genome.flat_gap_g[count]);
                count += 1;
            }
        }

        let syn_co: Vec<f64> = genome.leak_g.clone();
        let syn_int: Vec<f64> = genome.leak_g.clone();
        let gap_co: Vec<f64> = gap_g.iter().map(|line| line.iter().sum()).collect();
        let gap_int: Vec<f64> = genome.leak_g.clone();
        let leak_int: Vec<f64> = genome.leak_g.iter().zip(genome.leak_e.iter()).map(|(g, e)| g * e).collect();

        let leak_g = genome.leak_g;
        let leak_e = genome.leak_e;
        let gate_beta = genome.gate_beta;
        let gate_adjust = genome.gate_adjust;

        Network {
            syn_co,
            syn_int,
            gap_co,
            gap_int,
            leak_int,
            syn_indices,
            syn_g,
            syn_e,
            leak_g,
            leak_e,
            gate_beta,
            gate_adjust,
            gap_indices,
            gap_g,
        }

    }
}
