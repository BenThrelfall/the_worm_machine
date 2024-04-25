use std::{
    fs::File,
    io::{BufReader, BufWriter},
    sync::atomic::{AtomicU32, Ordering},
    time::Instant,
};

use itertools::Itertools;
use nalgebra::ComplexField;
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

    //Inserting Self Synapses
    //for i in 0..full_syn_g.len(){
    //    full_syn_g[i][i] = 100.0;
    //}

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let mut world = World::new();

    let file = File::open("results/full_evolutionv2_latest_population.json").unwrap();
    let buffer = BufReader::new(file);
    let mut population: Vec<Creature<Genome>> = serde_json::from_reader(buffer).unwrap();

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let mut prev_best: f64 = 100000.0;

    let mut heat = 0.3f64;
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
                let mut model = factory.build(creature.genome().clone());
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

        let file = File::create("results/full_evolutionv2_latest_population.json").unwrap();
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
    let (time_trace, mut full_syn_g, full_gap_g, full_syn_e, sensory_indices) = read_data();

    let file = File::open(
        "final_results/second_round_full_no_calc_gates_evolution_latest_population.json",
    )
    .unwrap();
    let buffer = BufReader::new(file);
    let genomes: Vec<Genome> = serde_json::from_reader(buffer).unwrap();

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

    println!("{}", specification.syn_len);
    println!("{}", syn_types.len());

    //let genome = genomes.first().unwrap();
    let genome = SmallGenome::default(syn_types);

    let multiplier = 15.0;
    let adjust = -12.0;

    let mut model = factory.build_with_calc_gates(genome.expand(&specification));

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let result = model.recorded_run_sensory(
        voltage,
        gates,
        0.0001,
        10.0,
        &time_trace,
        multiplier,
        adjust,
        &sensory_indices,
        10,
    );

    println!("{}", result.error);

    let file = File::create("results/tmp_testing_run_direct.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &result.volt_record).unwrap();
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

    println!("{}", specification.syn_len);
    println!("{}", syn_types.len());

    //let genome = genomes.first().unwrap();
    let genome = SmallGenome {
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

    let multiplier = 1.0;
    let adjust = 0.0;

    let mut expanded_genome = genome.expand(&specification);

    println!("before: {} {}", expanded_genome.flat_gap_g.len(), expanded_genome.flat_syn_g.len());

    expanded_genome.flat_gap_g = full_gap_g.iter().flatten().map(|x| *x).filter(|x| *x != 0.0).collect();
    expanded_genome.flat_syn_g = full_syn_g.iter().flatten().map(|x| *x).filter(|x| *x != 0.0).collect();

    println!("after: {} {}", expanded_genome.flat_gap_g.len(), expanded_genome.flat_syn_g.len());

    let mut model = factory.build_with_calc_gates(expanded_genome);

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| -5.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let (voltage, gates) = model.run(voltage, gates, 0.0001, 5.0);

    let equil = voltage.clone();

    let time_trace: Vec<Frame> = (0..50)
        .map(|x| Frame {
            time: 10.0 * x as f64,
            data: (0..280)
                .map(|i| {
                    if i == 187 || i == 188 {
                        100000000.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        })
        .collect();

    let sensory_indices = vec![187, 188];

    let result = model.recorded_run_sensory(
        voltage,
        gates,
        0.0001,
        20.0,
        &time_trace,
        multiplier,
        adjust,
        &sensory_indices,
        100,
    );

    println!("{}", result.error);

    let file = File::create("results/con2_plm_equil.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &equil).unwrap();

    let file = File::create("results/con2_plm_test_run.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &result.volt_record).unwrap();
}
