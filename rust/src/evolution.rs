use std::ops::{Mul, Neg};

use itertools::Itertools;
use rand::prelude::*;
use rand_pcg::Pcg32;

use crate::{data::Frame, factory::Specification, genetics::{Genome, SmallGenome, SynapseType}, neuron::Network};

const DEFAULT_VOLTAGE: f64 = 0.0;
const DEFAULT_GATE: f64 = 0.0;

pub struct World {
    rng: Pcg32,
}

//Normal Genome
impl World {
    pub fn new() -> Self {
        let rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        World { rng }
    }

    pub fn random_population(
        &mut self,
        specification: &Specification,
        count: usize,
    ) -> Vec<Genome> {
        (0..count)
            .map(|_| self.random_genome(specification))
            .collect()
    }

    fn random_genome(&mut self, specification: &Specification) -> Genome {
        let flat_syn_g = (0..specification.syn_len)
            .map(|_| self.rng.gen_range(0f64..100.0))
            .collect();
        let flat_syn_e = (0..specification.syn_len)
            .map(|_| self.rng.gen_range(-100f64..100.0))
            .collect();
        let flat_gap_g = (0..specification.gap_len)
            .map(|_| self.rng.gen_range(0f64..100.0))
            .collect();

        let gate_beta = (0..specification.model_len)
            .map(|_| self.rng.gen_range(0f64..1.0))
            .collect();
        let gate_adjust = (0..specification.model_len)
            .map(|_| self.rng.gen_range(-50f64..50.0))
            .collect();
        let leak_g = (0..specification.model_len)
            .map(|_| self.rng.gen_range(0f64..100.0))
            .collect();
        let leak_e = (0..specification.model_len)
            .map(|_| self.rng.gen_range(-100f64..100.0))
            .collect();

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

    pub fn selection(
        &mut self,
        population: &Vec<Genome>,
        results: &Vec<f64>,
        count: usize,
    ) -> Vec<Genome> {
        population
            .iter()
            .zip(results)
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|x| x.0.clone())
            .take(count)
            .collect()
    }

    pub fn crossover(&mut self, population: &mut Vec<Genome>) {
        let inital_len = population.len();

        for i in 0..inital_len {
            for j in i + 1..inital_len {
                if i == j {
                    continue;
                }
                let (a_child, b_child) = self.random_breed(&population[i], &population[j]);
                population.push(a_child);
                population.push(b_child);
            }
        }
    }

    pub fn mutate(
        &mut self,
        population: &mut Vec<Genome>,
        genome_rate: f64,
        dna_rate: f64,
        heat: f64,
    ) {
        for genome in population {
            if self.rng.gen_bool(genome_rate) {
                self.mutate_genome(genome, dna_rate, heat);
            }
        }
    }

    fn mutate_genome(&mut self, genome: &mut Genome, rate: f64, heat: f64) {
        self.mutate_vec(
            &mut genome.flat_gap_g,
            rate,
            -100.0,
            100.0,
            1.0,
            500.0,
            heat,
        );

        self.mutate_vec(
            &mut genome.flat_syn_g,
            rate,
            -100.0,
            100.0,
            1.0,
            500.0,
            heat,
        );
        self.mutate_vec(
            &mut genome.flat_syn_e,
            rate,
            -100.0,
            100.0,
            -500.0,
            500.0,
            heat,
        );

        self.mutate_vec(&mut genome.leak_g, rate, -100.0, 100.0, 1.0, 100.0, heat);
        self.mutate_vec(&mut genome.leak_e, rate, -100.0, 100.0, -500.0, 500.0, heat);

        self.mutate_vec(&mut genome.gate_beta, rate, -1.0, 1.0, 0.0, 10.0, heat);
        self.mutate_vec(
            &mut genome.gate_adjust,
            rate,
            -100.0,
            100.0,
            -100.0,
            100.0,
            heat,
        );
    }

    fn random_breed(&mut self, x_parent: &Genome, y_parent: &Genome) -> (Genome, Genome) {
        let mut a_child = Genome::new();
        let mut b_child = Genome::new();

        for i in 0..x_parent.flat_syn_g.len() {
            if self.rng.gen() {
                a_child.flat_syn_g.push(x_parent.flat_syn_g[i]);
                a_child.flat_syn_e.push(x_parent.flat_syn_e[i]);

                b_child.flat_syn_g.push(y_parent.flat_syn_g[i]);
                b_child.flat_syn_e.push(y_parent.flat_syn_e[i]);
            } else {
                b_child.flat_syn_g.push(x_parent.flat_syn_g[i]);
                b_child.flat_syn_e.push(x_parent.flat_syn_e[i]);

                a_child.flat_syn_g.push(y_parent.flat_syn_g[i]);
                a_child.flat_syn_e.push(y_parent.flat_syn_e[i]);
            }
        }

        for i in 0..x_parent.leak_g.len() {
            if self.rng.gen() {
                a_child.leak_g.push(x_parent.leak_g[i]);
                a_child.leak_e.push(x_parent.leak_e[i]);

                b_child.leak_g.push(y_parent.leak_g[i]);
                b_child.leak_e.push(y_parent.leak_e[i]);
            } else {
                b_child.leak_g.push(x_parent.leak_g[i]);
                b_child.leak_e.push(x_parent.leak_e[i]);

                a_child.leak_g.push(y_parent.leak_g[i]);
                a_child.leak_e.push(y_parent.leak_e[i]);
            }
        }

        for i in 0..x_parent.gate_beta.len() {
            if self.rng.gen() {
                a_child.gate_beta.push(x_parent.gate_beta[i]);
                a_child.gate_adjust.push(x_parent.gate_adjust[i]);

                b_child.gate_beta.push(y_parent.gate_beta[i]);
                b_child.gate_adjust.push(y_parent.gate_adjust[i]);
            } else {
                b_child.gate_beta.push(x_parent.gate_beta[i]);
                b_child.gate_adjust.push(x_parent.gate_adjust[i]);

                a_child.gate_beta.push(y_parent.gate_beta[i]);
                a_child.gate_adjust.push(y_parent.gate_adjust[i]);
            }
        }

        for i in 0..x_parent.flat_gap_g.len() {
            if self.rng.gen() {
                a_child.flat_gap_g.push(x_parent.flat_gap_g[i]);

                b_child.flat_gap_g.push(y_parent.flat_gap_g[i]);
            } else {
                b_child.flat_gap_g.push(x_parent.flat_gap_g[i]);

                a_child.flat_gap_g.push(y_parent.flat_gap_g[i]);
            }
        }

        (a_child, b_child)
    }
}

//Small Genome
impl World {

    pub fn small_random_population(
        &mut self,
        specification: &Specification,
        count: usize,
    ) -> Vec<SmallGenome> {
        (0..count)
            .map(|_| self.small_random_genome(specification))
            .collect()
    }

    pub fn small_random_genome(&mut self, specification: &Specification) -> SmallGenome {
        let syn_g = self.rng.gen_range(1f64..500f64);
        let syn_e_in = self.rng.gen_range(-100f64..100f64);
        let syn_e_ex = self.rng.gen_range(-100f64..100f64);
        let gap_g = self.rng.gen_range(1f64..500f64);
        let gate_beta = self.rng.gen_range(-2f64..2f64);
        let gate_adjust = self.rng.gen_range(-70f64..70f64);
        let leak_g = self.rng.gen_range(1f64..500f64);
        let leak_e = self.rng.gen_range(-100f64..100f64);

        let rate = self.rng.gen_range(0.01f64..1f64);
        let syn_types = (0..specification.syn_len)
            .map(|_| {
                if self.rng.gen_bool(rate) {
                    SynapseType::Excitatory
                } else {
                    SynapseType::Inhibitory
                }
            })
            .collect();

        SmallGenome {
            syn_g,
            syn_e_in,
            syn_e_ex,
            syn_types,
            gap_g,
            gate_beta,
            gate_adjust,
            leak_g,
            leak_e,
        }
    }

    pub fn small_mutate(
        &mut self,
        population: &mut Vec<SmallGenome>,
        genome_rate: f64,
        dna_rate: f64,
        heat: f64,
    ) {
        for genome in population.iter_mut().skip(0) {
            if self.rng.gen_bool(genome_rate) {
                self.small_mutate_genome(genome, dna_rate, heat);
            }
        }
    }

    fn small_mutate_genome(&mut self, genome: &mut SmallGenome, rate: f64, heat: f64) {

        self.mutate_value(&mut genome.syn_g, rate, -100.0, 100.0, 0.1, 2000.0, heat);
        self.mutate_value(&mut genome.syn_e_ex, rate, -100.0, 100.0, -200.0, 200.0, heat);
        self.mutate_value(&mut genome.syn_e_in, rate, -100.0, 100.0, -200.0, 200.0, heat);
        self.mutate_value(&mut genome.gap_g, rate, -100.0, 100.0, 0.1, 2000.0, heat);
        self.mutate_value(&mut genome.leak_g, rate, -100.0, 100.0, 0.1, 2000.0, heat);
        self.mutate_value(&mut genome.leak_e, rate, -100.0, 100.0, -200.0, 200.0, heat);
        self.mutate_value(&mut genome.gate_beta, rate, -10.0, 10.0, -10.0, 10.0, heat);

        genome.syn_types.iter_mut().for_each(|x| {
            if self.rng.gen_bool(rate * heat){
                *x = match x{
                    SynapseType::Excitatory => SynapseType::Inhibitory,
                    SynapseType::Inhibitory => SynapseType::Excitatory,
                }
            }
        });
    }

    pub fn small_crossover(&mut self, population: &mut Vec<SmallGenome>) {

        let inital_len = population.len();

        for i in 0..inital_len {
            for j in i + 1..inital_len {
                if i == j {
                    continue;
                }
                let (a_child, b_child) = self.small_random_breed(&population[i], &population[j]);
                population.push(a_child);
                population.push(b_child);
            }
        }
    }

    fn small_random_breed(&mut self, x_parent: &SmallGenome, y_parent: &SmallGenome) -> (SmallGenome, SmallGenome) {
        let mut a_child = SmallGenome::new();
        let mut b_child = SmallGenome::new();

        if self.rng.gen(){
            a_child.syn_g = x_parent.syn_g;
            b_child.syn_g = y_parent.syn_g;
        }
        else{
            b_child.syn_g = x_parent.syn_g;
            a_child.syn_g = y_parent.syn_g;
        }
        
        if self.rng.gen(){
            a_child.syn_e_in = x_parent.syn_e_in;
            b_child.syn_e_in = y_parent.syn_e_in;
        }
        else{
            b_child.syn_e_in = x_parent.syn_e_in;
            a_child.syn_e_in = y_parent.syn_e_in;
        }

        if self.rng.gen(){
            a_child.syn_e_ex = x_parent.syn_e_ex;
            b_child.syn_e_ex = y_parent.syn_e_ex;
        }
        else{
            b_child.syn_e_ex = x_parent.syn_e_ex;
            a_child.syn_e_ex = y_parent.syn_e_ex;
        }

        if self.rng.gen(){
            a_child.gap_g = x_parent.gap_g;
            b_child.gap_g = y_parent.gap_g;
        }
        else{
            b_child.gap_g = x_parent.gap_g;
            a_child.gap_g = y_parent.gap_g;
        }


        if self.rng.gen(){
            a_child.leak_g = x_parent.leak_g;
            b_child.leak_g = y_parent.leak_g;
        }
        else{
            b_child.leak_g = x_parent.leak_g;
            a_child.leak_g = y_parent.leak_g;
        }


        if self.rng.gen(){
            a_child.leak_e = x_parent.leak_e;
            b_child.leak_e = y_parent.leak_e;
        }
        else{
            b_child.leak_e = x_parent.leak_e;
            a_child.leak_e = y_parent.leak_e;
        }


        if self.rng.gen(){
            a_child.gate_beta = x_parent.gate_beta;
            b_child.gate_beta = y_parent.gate_beta;
        }
        else{
            b_child.gate_beta = x_parent.gate_beta;
            a_child.gate_beta = y_parent.gate_beta;
        }

        for i in 0..x_parent.syn_types.len() {
            if self.rng.gen() {
                a_child.syn_types.push(x_parent.syn_types[i]);

                b_child.syn_types.push(y_parent.syn_types[i]);
            } else {
                b_child.syn_types.push(x_parent.syn_types[i]);

                a_child.syn_types.push(y_parent.syn_types[i]);
            }
        }

        (a_child, b_child)
    }
}

//General Functions
impl World {
    fn mutate_vec(
        &mut self,
        vec: &mut Vec<f64>,
        rate: f64,
        delta_min: f64,
        delta_max: f64,
        bound_min: f64,
        bound_max: f64,
        heat: f64,
    ) {
        for item in vec {
            self.mutate_value(item, rate, delta_min, delta_max, bound_min, bound_max, heat);
        }
    }

    fn mutate_value(
        &mut self,
        item: &mut f64,
        rate: f64,
        delta_min: f64,
        delta_max: f64,
        bound_min: f64,
        bound_max: f64,
        heat: f64,
    ) {
        if self.rng.gen_bool(rate) {
            *item += self.rng.gen_range(delta_min..delta_max) * heat;
            *item = item.clamp(bound_min, bound_max);
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

    for i in 0..60 {
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

pub fn record_with_data(
    model: &mut Network,
    start_index: usize,
    data: &Vec<Frame>,
) -> Vec<Vec<f64>> {
    let model_size = model.leak_g.len();
    let mut voltage: Vec<f64> = (0..model_size).map(|_| DEFAULT_VOLTAGE).collect();
    let mut gates: Vec<f64> = (0..model_size).map(|_| DEFAULT_GATE).collect();

    let mut output: Vec<Vec<f64>> = Vec::new();
    let mut volt_record;

    for i in 0..15 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;
        let points = data[start_index + i].data.clone();

        voltage.splice(..points.len(), points);

        (voltage, gates, volt_record, _) = model.recorded_run(voltage, gates, 0.0001, runtime, 10);
        output.append(&mut volt_record);
    }

    let start_index = start_index + 15;

    for i in 0..60 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;

        (voltage, gates, volt_record, _) = model.recorded_run(voltage, gates, 0.0001, runtime, 10);
        output.append(&mut volt_record);
    }

    return output;
}

pub fn evaluate_std(model: &mut Network, start_index: usize, data: &Vec<Frame>) -> f64 {
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

    for i in 0..5 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;

        (voltage, gates) = model.run(voltage, gates, 0.01, runtime);
    }

    let start_index = start_index + 5;
    let mut error = 0f64;
    let mut memory = Vec::new();

    for i in 0..20 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;

        (voltage, gates) = model.run(voltage, gates, 0.01, runtime);

        memory.push(voltage.clone());
    }

    for i in 0..voltage.len() {
        let mean = memory.iter().map(|frame| frame[i]).sum::<f64>() / memory.len() as f64;
        error += (memory
            .iter()
            .map(|frame| (frame[i] - mean).powf(2.0))
            .sum::<f64>()
            / memory.len() as f64)
            .mul(0.1)
            .neg()
            .exp();
    }

    return error;
}

pub fn predict(
    model: &mut Network,
    start_index: usize,
    data: &Vec<Frame>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

    let mut predicted = Vec::new();
    let mut actual = Vec::new();

    for i in 0..15 {
        let runtime = data[start_index + i + 1].time - data[start_index + i].time;
        let points = data[start_index + i].data.clone();

        (voltage, gates) = model.run(voltage, gates, 0.01, runtime);

        predicted.push(voltage.clone());
        actual.push(points.clone());
    }

    return (predicted, actual);
}
