use rand_pcg::Pcg32;
use rand::prelude::*;

use crate::{factory::Specification, neuron::Network, Frame};

const DEFAULT_VOLTAGE: f64 = 0.0;
const DEFAULT_GATE: f64 = 0.0;

#[derive(Clone)]
pub struct Genome {
    pub flat_syn_g: Vec<f64>,
    pub flat_syn_e: Vec<f64>,
    pub flat_gap_g: Vec<f64>,
    pub gate_beta: Vec<f64>,
    pub gate_adjust: Vec<f64>,
    pub leak_g: Vec<f64>,
    pub leak_e: Vec<f64>,
}

pub struct World {
    rng: Pcg32,
}

impl World {

    pub fn new() -> Self{
        let rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        World{
            rng
        }
    }

    pub fn random_population(&mut self, specification: &Specification, count: usize) -> Vec<Genome> {
        (0..count).map(|_| self.random_genome(specification)).collect()
    }

    fn random_genome(&mut self, specification: &Specification) -> Genome{

        let flat_syn_g = (0..specification.syn_len).map(|_| self.rng.gen_range(0f64..100.0)).collect();
        let flat_syn_e = (0..specification.syn_len).map(|_| self.rng.gen_range(-100f64..100.0)).collect();
        let flat_gap_g = (0..specification.gap_len).map(|_| self.rng.gen_range(0f64..100.0)).collect();

        let gate_beta = (0..specification.model_len).map(|_| self.rng.gen_range(0f64..1.0)).collect();
        let gate_adjust = (0..specification.model_len).map(|_| self.rng.gen_range(-50f64..50.0)).collect();
        let leak_g = (0..specification.model_len).map(|_| self.rng.gen_range(0f64..100.0)).collect();
        let leak_e = (0..specification.model_len).map(|_| self.rng.gen_range(-100f64..100.0)).collect();

        Genome{
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

pub fn evaluate(model: &mut Network, start_index: usize, data: &Vec<Frame>) -> f64 {
    let model_size = model.leak_g.len();
    let mut voltage: Vec<f64> = (0..model_size).map(|_| DEFAULT_VOLTAGE).collect();
    let mut gates: Vec<f64> = (0..model_size).map(|_| DEFAULT_GATE).collect();

    for i in 0..15 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;
        let points = data[start_index + i].data.clone();

        voltage.splice(..points.len(), points);

        (voltage, gates) = model.run(voltage, gates, 0.01, runtime);
    }

    let start_index = start_index + 15;
    let mut error = 0f64;

    for i in 0..15 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;
        let points = data[start_index + i].data.clone();

        (voltage, gates) = model.run(voltage, gates, 0.01, runtime);

        error += voltage
            .iter()
            .zip(points)
            .map(|(volt, point)| (volt - point).powf(2.0))
            .sum::<f64>();
    }

    return error;
}
