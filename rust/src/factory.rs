use std::{fs::File, io::BufWriter};

use crate::{
    genetics::{calculate_gate_adjust, Genome},
    neuron::Network,
};

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
    pub fn new(full_syn_g: &Vec<Vec<f64>>, full_gap_g: &Vec<Vec<f64>>) -> Self {
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

    pub fn build_with_calc_gates(&self, genome: Genome) -> Network {

        let mut full_syn_g: Vec<Vec<f64>> = (0..self.model_len)
            .map(|_| (0..self.model_len).map(|_| 0.0).collect())
            .collect();

        let mut full_syn_e: Vec<Vec<f64>> = (0..self.model_len)
            .map(|_| (0..self.model_len).map(|_| 0.0).collect())
            .collect();

        let mut full_gap_g: Vec<Vec<f64>> = (0..self.model_len)
            .map(|_| (0..self.model_len).map(|_| 0.0).collect())
            .collect();

        let mut model = self.build(genome);

        for row in 0..self.model_len{
            for compact_col in 0..model.syn_g[row].len(){
                let full_col = model.syn_indices[row][compact_col];
                full_syn_g[row][full_col] = model.syn_g[row][compact_col];
                full_syn_e[row][full_col] = model.syn_e[row][compact_col];
            }
        }

        for row in 0..self.model_len{
            for compact_col in 0..model.gap_g[row].len(){
                let full_col = model.gap_indices[row][compact_col];
                full_gap_g[row][full_col] = model.gap_g[row][compact_col];
            }
        }

        let file = File::create("processed_data/proc_calc_syn_g.json").unwrap();
        let buffer = BufWriter::new(file);
        serde_json::to_writer(buffer, &(full_syn_e.iter().flatten().map(|x| *x).collect::<Vec<f64>>())).unwrap();

        let calc_gates = calculate_gate_adjust(&model.leak_g, &model.leak_e, &full_syn_g, &full_syn_e, &full_gap_g);

        model.gate_adjust = calc_gates;

        return model;
    }

    pub fn build(&self, genome: Genome) -> Network {
        let syn_indices = self.syn_indices.clone();
        let gap_indices = self.gap_indices.clone();

        let mut syn_g = Vec::new();
        let mut syn_e = Vec::new();

        let mut count = 0;
        for (i, line) in syn_indices.iter().enumerate() {
            syn_g.push(Vec::new());
            syn_e.push(Vec::new());

            for _ in line.iter() {
                syn_g[i].push(genome.flat_syn_g[count]);
                syn_e[i].push(genome.flat_syn_e[count]);
                count += 1;
            }
        }

        let mut gap_g = Vec::new();

        let mut count = 0;
        for (i, line) in gap_indices.iter().enumerate() {
            gap_g.push(Vec::new());

            for _ in line.iter() {
                gap_g[i].push(genome.flat_gap_g[count]);
                count += 1;
            }
        }

        let syn_co: Vec<f64> = genome.leak_g.clone();
        let syn_int: Vec<f64> = genome.leak_g.clone();
        let gap_co: Vec<f64> = gap_g.iter().map(|line| line.iter().sum()).collect();
        let gap_int: Vec<f64> = genome.leak_g.clone();
        let leak_int: Vec<f64> = genome
            .leak_g
            .iter()
            .zip(genome.leak_e.iter())
            .map(|(g, e)| g * e)
            .collect();

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

#[cfg(test)]
mod tests {
    use crate::{genetics::{calculate_gate_adjust, Genome, SmallGenome, SynapseType}, data::read_data};

    use super::Factory;

    #[test]
    fn small_build_test() {
        let (_, full_syn_g, full_gap_g, full_syn_e, _) = read_data();

        let leak_g = (0..280).map(|_| 10f64).collect();
        let leak_e = (0..280).map(|_| -35f64).collect();
        let gate_beta = (0..280).map(|_| 0.125f64).collect();
        let gate_adjust = (0..280).map(|_| -15f64).collect();

        let syn_types : Vec<SynapseType> = full_syn_e.iter().flatten().zip(full_syn_g.iter().flatten()).filter(|(_, g)| **g != 0.0).map(|(e, _)| if *e == 0.0{
            SynapseType::Excitatory
        }else{
            SynapseType::Inhibitory
        }).collect();

        let default_genome = SmallGenome{
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

        let flat_gap_g : Vec<f64> = full_gap_g.iter().flatten().filter(|x| **x != 0.0).map(|x| *x).collect();
        let mut flat_syn_g = Vec::new();
        let mut flat_syn_e = Vec::new();

        for row in 0..full_syn_g.len(){
            for col in 0..full_syn_g[row].len(){
                if full_syn_g[row][col] != 0.0{
                    flat_syn_g.push(full_syn_g[row][col]);
                    flat_syn_e.push(full_syn_e[row][col]);
                }
            }
        }

        let big_genome = Genome{
            flat_syn_g,
            flat_syn_e,
            flat_gap_g,
            gate_beta,
            gate_adjust,
            leak_g,
            leak_e,
        };

        let factory = Factory::new(&full_syn_g, &full_gap_g);
        let specification = factory.get_specification();

        let model = factory.build(default_genome.expand(&specification));
        let big_model = factory.build(big_genome);

        assert_eq!(model.leak_g, big_model.leak_g);
        assert_eq!(model.gate_adjust, big_model.gate_adjust);
        assert_eq!(model.gap_g, big_model.gap_g);
        assert_eq!(model.syn_g, big_model.syn_g);
        assert_eq!(model.syn_e, big_model.syn_e);

    }


    #[test]
    fn gate_calc_build_test() {
        let (_, full_syn_g, full_gap_g, full_syn_e, _) = read_data();

        let leak_g = (0..280).map(|_| 10f64).collect();
        let leak_e = (0..280).map(|_| -35f64).collect();
    
        let gate_calc = calculate_gate_adjust(&leak_g, &leak_e, &full_syn_g, &full_syn_e, &full_gap_g);

        let syn_types : Vec<SynapseType> = full_syn_e.iter().flatten().zip(full_syn_g.iter().flatten()).filter(|(_, g)| **g != 0.0).map(|(e, _)| if *e == 0.0{
            SynapseType::Excitatory
        }else{
            SynapseType::Inhibitory
        }).collect();

        let default_genome = SmallGenome{
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

        let factory = Factory::new(&full_syn_g, &full_gap_g);
        let specification = factory.get_specification();

        let model = factory.build_with_calc_gates(default_genome.expand(&specification));

        assert_eq!(model.leak_g, leak_g);
        assert_eq!(model.gate_adjust, gate_calc);
    }
}

