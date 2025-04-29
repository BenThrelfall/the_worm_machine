use std::{
    fs::File,
    io::{BufReader, BufWriter},
    sync::atomic::{AtomicU32, Ordering},
    time::Instant,
};

use itertools::Itertools;
use rand::Rng;
use rayon::prelude::*;

use crate::{
    data::{read_data, Frame},
    evolution::{Creature, World},
    factory::Factory,
    genetics::{Genome, SmallGenome, SynapseType},
};

pub fn evolutionary_training() {
    let (time_trace, full_syn_g, full_gap_g, _full_syn_e, sensory_indices) = read_data();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let mut world = World::new();

    let mut population: Vec<Creature<Genome>> = world
        .random_population(&specification, 200)
        .into_iter()
        .map(|x| Creature::new(x))
        .collect();

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let mut prev_best: f64 = 100000.0;

    let mut heat = 0.2f64;
    let mut genome_rate = 0.5;
    let mut gene_rate = 0.05;
    let mut breed_rate = 0.5;

    let adult_size = 50;

    let mut no_prog_counter = 0;

    for _ in 0..2000 {
        let now = Instant::now();

        let atomic_counter = AtomicU32::new(0);

        population
            .par_iter_mut()
            .filter(|creature| creature.error().is_none())
            .for_each(|creature| {
                let mut model = factory.build_with_calc_gates(creature.genome().clone());
                let result = model
                    .recorded_run_sensory(
                        voltage.clone(),
                        gates.clone(),
                        0.01,
                        300.0,
                        &time_trace,
                        15.0,
                        -10.0,
                        &sensory_indices,
                        10000,
                    )
                    .error;

                atomic_counter.fetch_add(1, Ordering::Relaxed);
                creature.eval(result);
            });

        population = population
            .into_iter()
            .sorted_by(|a, b| a.error().partial_cmp(&b.error()).unwrap())
            .take(adult_size)
            .collect();

        let best = population.first().unwrap().error().unwrap();

        let file = File::create("fullv2_calc.json").unwrap();
        let buffer = BufWriter::new(file);
        serde_json::to_writer(buffer, &population).unwrap();

        world.crossover(&mut population, 200 - adult_size, breed_rate);

        world.mutate(&mut population, genome_rate, gene_rate, heat, adult_size);

        if (prev_best - best) < 0.01 {
            no_prog_counter += 1;
        } else {
            no_prog_counter = 0;
        }

        if no_prog_counter > 15 {
            breed_rate = (breed_rate + world.rng.gen_range(-0.05..0.05)).clamp(0.01, 1.0);
            heat = (heat + world.rng.gen_range(-0.05..0.05)).clamp(0.01, 1.0);
            genome_rate = (genome_rate + world.rng.gen_range(-0.05..0.05)).clamp(0.01, 1.0);
            gene_rate = (gene_rate + world.rng.gen_range(-0.02..0.01)).clamp(0.01, 1.0);
        }

        let elapsed = now.elapsed();

        println!(
            "Best: {:.3} d {:.3} d% {:.3} (heat {}; genome rate {}; dna rate {}) (time {:.3?}) (pop: {} Sim: {})",
            best,
            prev_best - best,
            (prev_best - best) / prev_best,
            heat,
            genome_rate,
            gene_rate,
            elapsed,
            population.len(),
            atomic_counter.load(Ordering::Relaxed),
        );
        prev_best = best;
    }
}

pub fn gate_calculation() {
    let (time_trace, full_syn_g, full_gap_g, _full_syn_e, sensory_indices) = read_data();

    //Inserting Self Synapses
    //for i in 0..full_syn_g.len() {
    //    full_syn_g[i][i] = 100.0;
    //}

    let file = File::open("final_results/final_full_no_calc.json").unwrap();
    let buffer = BufReader::new(file);
    let creatures: Vec<Creature<Genome>> = serde_json::from_reader(buffer).unwrap();

    println!("{}", creatures[0].error().unwrap());

    let genome = creatures.first().unwrap().genome().clone();

    let mut params = 0;

    params += genome.flat_gap_g.len();

    params += genome.flat_syn_g.len();
    params += genome.flat_syn_e.len();

    params += genome.leak_e.len();
    params += genome.leak_g.len();

    params += genome.gate_adjust.len();
    params += genome.gate_beta.len();

    println!("Params: {}", params);


    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let multiplier = 15.0;
    let adjust = -10.0;

    let mut model = factory.build(genome);

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let result = model.extra_recorded_run_sensory(
        voltage,
        gates,
        0.01,
        300.0,
        &time_trace,
        multiplier,
        adjust,
        &sensory_indices,
        10,
    );

    println!("{}", result.error);
    println!("Evals: {}", result.evals_performed);
    println!("Raw:");
    println!("mse: {}; abs error: {}", result.raw_mse, result.raw_mabse);
    println!("Proc:");
    println!("mse: {}; abs error: {}", result.proc_mse, result.proc_mabse);

    let file = File::create("results/full_test_run.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &result.volt_record).unwrap();

    let file = File::create("results/full_test_run_evals.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &result.neuron_evals).unwrap();
}

pub fn plm_test_run() {
    let (_, full_syn_g, full_gap_g, full_syn_e, _) = read_data();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let syn_types: Vec<SynapseType> = full_syn_e
        .iter()
        .flatten()
        .zip(full_syn_g.iter().flatten())
        .filter(|(_, g)| **g != 0.0)
        .map(|(e, _)| {
            if *e == 0.0 {
                SynapseType::Excitatory
            } else {
                SynapseType::Inhibitory
            }
        })
        .collect();

    //let genome = genomes.first().unwrap();
    let genome = SmallGenome::default(syn_types);

    let expanded_genome = genome.expand(&specification);

    let mut model = factory.build_with_calc_gates(expanded_genome);

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| -60.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    //let (voltage, gates) = model.run(voltage, gates, 0.0001, 5.0);

    let equil = voltage.clone();

    let time_trace: Vec<Frame> = (0..50)
        .map(|x| Frame {
            time: 30.0 * x as f64,
            data: (0..280).map(|i| if i == 128 { 0.0 } else { 0.0 }).collect(),
        })
        .collect();

    let sensory_indices = vec![];

    let result = model.recorded_run_sensory(
        voltage,
        gates,
        0.01,
        8.0,
        &time_trace,
        1.0,
        0.0,
        &sensory_indices,
        1,
    );

    println!("{}", result.error);

    let file = File::create("final_results/tmp_be.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &equil).unwrap();

    let file = File::create("final_results/tmp_b.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &result.volt_record).unwrap();
}

pub fn full_patch_clamping() {
    let (_, full_syn_g, full_gap_g, full_syn_e, _) = read_data();

    let syn_types: Vec<SynapseType> = full_syn_e
        .iter()
        .flatten()
        .zip(full_syn_g.iter().flatten())
        .filter(|(_, g)| **g != 0.0)
        .map(|(e, _)| {
            if *e == 0.0 {
                SynapseType::Excitatory
            } else {
                SynapseType::Inhibitory
            }
        })
        .collect();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let genome = SmallGenome::default(syn_types);

    let mut model = factory.build_with_calc_gates(genome.expand(&specification));

    let mut results = Vec::new();

    for i in 0..specification.model_len {
        results.push(Vec::new());

        for v in (-150..150).step_by(5) {
            let voltage: Vec<f64> = (0..specification.model_len)
                .map(|n| if n == i { v as f64 } else { -5.0 })
                .collect();
            let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

            let (voltage, _gates, internal_input) = model.run_clamped(
                voltage,
                gates,
                (0..specification.model_len).map(|n| n == i).collect(),
                0.01,
                20.0,
            );

            results[i].push((voltage[i], internal_input[i]));
        }
    }

    println!("");

    let file = File::create("results/cook_full_clamp.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &results).unwrap();
}

pub fn two_node_patch_clamping() {
    let full_syn_g = vec![vec![0.0, 100.0], vec![100.0, 0.0]];

    let full_gap_g = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

    let syn_types: Vec<SynapseType> = vec![SynapseType::Excitatory, SynapseType::Excitatory];

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let mut results = Vec::new();

    for syn_n in 0..20 {
        results.push(Vec::new());
        let mut genome = SmallGenome::default(syn_types.clone()).expand(&specification);
        genome.flat_syn_g = vec![100.0, syn_n as f64 * 100.0];
        let mut model = factory.build_with_calc_gates(genome);

        for v in (-150..150).step_by(5) {
            let voltage: Vec<f64> = (0..specification.model_len)
                .map(|n| if n == 0 { v as f64 } else { -5.0 })
                .collect();
            let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

            let (voltage, _gates, internal_input) = model.run_clamped(
                voltage,
                gates,
                (0..specification.model_len).map(|n| n == 0).collect(),
                0.01,
                60.0,
            );

            results[syn_n].push((voltage[0], internal_input[0]));
        }
    }

    println!("");

    let file = File::create("results/two_node_a2b_clamp.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &results).unwrap();
}
