use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::factory::Specification;

#[derive(Clone, Serialize, Deserialize)]
pub struct Genome {
    pub flat_syn_g: Vec<f64>,
    pub flat_syn_e: Vec<f64>,
    pub flat_gap_g: Vec<f64>,
    pub gate_beta: Vec<f64>,
    pub gate_adjust: Vec<f64>,
    pub leak_g: Vec<f64>,
    pub leak_e: Vec<f64>,
}

impl Genome {
    pub fn new() -> Self {
        Genome {
            flat_syn_g: Vec::new(),
            flat_syn_e: Vec::new(),
            flat_gap_g: Vec::new(),
            gate_beta: Vec::new(),
            gate_adjust: Vec::new(),
            leak_g: Vec::new(),
            leak_e: Vec::new(),
        }
    }
}

pub fn calculate_gate_adjust(
    leak_g: &Vec<f64>,
    leak_e: &Vec<f64>,
    full_syn_g: &Vec<Vec<f64>>,
    full_syn_e: &Vec<Vec<f64>>,
    full_gap_g: &Vec<Vec<f64>>,
) -> Vec<f64> {
    let length = leak_g.len();

    let matrix_iter = (0..length)
        .map(|i| {
            (0..length).map(move |j| {
                if i != j {
                    //TODO: make sure this is the correct order for selecting variables
                    -(1.0 / leak_g[i]) * full_gap_g[i][j]
                } else {
                    1.0 + (1.0 / leak_g[i])
                        * (0..length)
                            .map(|k| full_gap_g[i][k] + (full_syn_g[i][k] / 2.0))
                            .sum::<f64>()
                }
            })
        })
        .flatten();

    let matrix = DMatrix::from_row_iterator(length, length, matrix_iter);

    let inverse_mat = matrix.try_inverse().unwrap();

    let vector_iter = (0..length).map(|i| {
        leak_e[i]
            + (1.0 / leak_g[i])
                * (0..length)
                    .map(|j| (full_syn_e[i][j] * full_syn_g[i][j]) / 2.0)
                    .sum::<f64>()
    });

    let vector = DVector::from_iterator(length, vector_iter);

    let solution = inverse_mat * vector;

    solution.iter().map(|n| *n).collect()
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SmallGenome {
    pub syn_g: f64,
    pub syn_e_in: f64,
    pub syn_e_ex: f64,
    pub syn_types: Vec<SynapseType>,
    pub gap_g: f64,
    pub gate_beta: f64,
    pub gate_adjust: f64,
    pub leak_g: f64,
    pub leak_e: f64,
}

impl SmallGenome {
    pub fn new() -> SmallGenome {
        SmallGenome {
            syn_g: 0.0,
            syn_e_in: 0.0,
            syn_e_ex: 0.0,
            syn_types: Vec::new(),
            gap_g: 0.0,
            gate_beta: 0.0,
            gate_adjust: 0.0,
            leak_g: 0.0,
            leak_e: 0.0,
        }
    }

    pub fn null_model(syn_types : Vec<SynapseType>) -> Self{
        SmallGenome{
            syn_g: 0.0001,
            syn_e_in: -0.0,
            syn_e_ex: 0.0,
            syn_types: syn_types,
            gap_g: 0.0001,
            gate_beta: 0.125,
            gate_adjust: 0.0,
            leak_g: 0.26,
            leak_e: -8.8,
        }
    }

    pub fn default(syn_types : Vec<SynapseType>) -> Self{
        SmallGenome{
            syn_g: 100.0,
            syn_e_in: -45.0,
            syn_e_ex: 0.0,
            syn_types,
            gap_g: 100.0,
            gate_beta: 0.125,
            gate_adjust: -15.0,
            leak_g: 10.0,
            leak_e: -35.0,
        }
    }

    pub fn expand(&self, specification: &Specification) -> Genome {
        let flat_syn_g = self.syn_types.iter().map(|_| self.syn_g).collect();
        let flat_syn_e = self
            .syn_types
            .iter()
            .map(|t| match t {
                SynapseType::Excitatory => self.syn_e_ex,
                SynapseType::Inhibitory => self.syn_e_in,
            })
            .collect();

        let flat_gap_g = (0..specification.gap_len).map(|_| self.gap_g).collect();

        let gate_beta = (0..specification.model_len)
            .map(|_| self.gate_beta)
            .collect();
        let gate_adjust = (0..specification.model_len)
            .map(|_| self.gate_adjust)
            .collect();

        let leak_g = (0..specification.model_len).map(|_| self.leak_g).collect();
        let leak_e = (0..specification.model_len).map(|_| self.leak_e).collect();

        Genome {
            flat_syn_g,
            flat_syn_e,
            flat_gap_g,
            gate_beta,
            gate_adjust,
            leak_g,
            leak_e,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::calculate_gate_adjust;

    #[test]
    fn gate_calc_test() {
        let leak_g = vec![10.0, 15.0, 8.0];
        let leak_e = vec![-35.0, -20.0, -40.0];

        let full_syn_g = vec![
            vec![0.0, 90.0, 100.0],
            vec![50.0, 0.0, 0.0],
            vec![0.0, 70.0, 0.0],
        ];

        let full_syn_e = vec![
            vec![0.0, -40.0, 0.0],
            vec![-50.0, 0.0, 0.0],
            vec![0.0, -30.0, 0.0],
        ];

        let full_gap_g = vec![
            vec![0.0, 90.0, 0.0],
            vec![90.0, 0.0, 150.0],
            vec![0.0, 150.0, 0.0],
        ];

        let result = calculate_gate_adjust(&leak_g, &leak_e, &full_syn_g, &full_syn_e, &full_gap_g);

        let expected = vec![-24.6849, -29.595, -30.0997];

        for i in 0..3 {
            assert!((result[i] - expected[i]).abs() < 0.001);
        }
    }
}
