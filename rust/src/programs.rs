use std::{
    fs::File,
    io::{BufReader, BufWriter},
    time::Instant,
};

use itertools::Itertools;
use rayon::prelude::*;

use crate::{
    data::read_data,
    evolution::World,
    factory::Factory,
    genetics::{Genome, SmallGenome, SynapseType},
};

pub fn evolutionary_training() {
    let (time_trace, full_syn_g, full_gap_g, _full_syn_e, sensory_indices) = read_data();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let mut world = World::new();

    let file = File::open("results/full_evolution_latest_population.json").unwrap();
    let buffer = BufReader::new(file);
    let mut population: Vec<Genome> = serde_json::from_reader(buffer).unwrap();

    let mut heat = 0.3f64;

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let mut prev_best: f64 = 100000.0;

    let mut genome_rate = 0.99;
    let mut gene_rate = 0.99;

    let mut no_prog_count = 0;

    for _ in 0..2000 {
        let now = Instant::now();

        let results: Vec<f64> = population
            .par_iter_mut()
            .map(|genome| factory.build_with_calc_gates(genome.clone()))
            .map(|mut model| {
                model
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
                    .error
            })
            .collect();

        let selection: Vec<(&Genome, f64)> = population
            .iter()
            .zip(results)
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .take(10)
            .collect();

        let best = selection.first().unwrap().1;

        population = selection.iter().map(|x| x.0.clone()).collect();

        let file = File::create("results/full_evolution_latest_population.json").unwrap();
        let buffer = BufWriter::new(file);
        serde_json::to_writer(buffer, &population).unwrap();

        world.crossover(&mut population);

        population.push(population.first().unwrap().clone());

        world.mutate(&mut population, genome_rate, gene_rate, heat);

        if (prev_best - best).abs() < 0.01{
            no_prog_count += 1;
        }
        else{
            no_prog_count = 0;
        }

        if no_prog_count >= 3{
            no_prog_count = 0;

            if heat == 0.3 && gene_rate == 0.99{
                gene_rate = 0.5;
                genome_rate = 0.5;
            }
            else if heat == 0.3 && gene_rate == 0.5{
                heat = 0.1;
                gene_rate = 0.99;
                genome_rate = 0.99;
            }
            else if heat == 0.1 && gene_rate == 0.99{
                gene_rate = 0.5;
                genome_rate = 0.5;
            }
            else if heat == 0.1 && gene_rate == 0.5{
                heat = 0.01;
                gene_rate = 0.99;
                genome_rate = 0.99;
            }
            else if heat == 0.01 && gene_rate == 0.99{
                gene_rate = 0.5;
                genome_rate = 0.5;
            }
            else{
                heat = 0.3;
                gene_rate = 0.99;
                genome_rate = 0.99;
            }

        }

        let elapsed = now.elapsed();

        println!(
            "Best: {:.3} d {:.3} d% {:.3} (heat {}; genome rate {}; dna rate {}) (time {:.3?})",
            best,
            prev_best - best,
            (prev_best - best) / prev_best,
            heat,
            genome_rate,
            gene_rate,
            elapsed
        );
        prev_best = best;
    }
}

pub fn evolutionary_training_no_calc_gates() {
    let (time_trace, full_syn_g, full_gap_g, _full_syn_e, sensory_indices) = read_data();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let mut world = World::new();

    let file = File::open("results/full_no_calc_gates_evolution_latest_population.json").unwrap();
    let buffer = BufReader::new(file);
    let mut population: Vec<Genome> = serde_json::from_reader(buffer).unwrap();

    let mut heat = 0.3f64;

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let mut prev_best: f64 = 100000.0;

    let mut genome_rate = 0.99;
    let mut gene_rate = 0.99;

    let mut no_prog_count = 0;

    for _ in 0..2000 {
        let now = Instant::now();

        let results: Vec<f64> = population
            .par_iter_mut()
            .map(|genome| factory.build(genome.clone()))
            .map(|mut model| {
                model
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
                    .error
            })
            .collect();

        let selection: Vec<(&Genome, f64)> = population
            .iter()
            .zip(results)
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .take(10)
            .collect();

        let best = selection.first().unwrap().1;

        population = selection.iter().map(|x| x.0.clone()).collect();

        let file = File::create("results/full_no_calc_gates_evolution_latest_population.json").unwrap();
        let buffer = BufWriter::new(file);
        serde_json::to_writer(buffer, &population).unwrap();

        world.crossover(&mut population);

        population.push(population.first().unwrap().clone());

        world.mutate(&mut population, genome_rate, gene_rate, heat);

        if (prev_best - best).abs() < 0.01{
            no_prog_count += 1;
        }
        else{
            no_prog_count = 0;
        }

        if no_prog_count >= 3{
            no_prog_count = 0;

            if heat == 0.3 && gene_rate == 0.99{
                gene_rate = 0.5;
                genome_rate = 0.5;
            }
            else if heat == 0.3 && gene_rate == 0.5{
                heat = 0.1;
                gene_rate = 0.99;
                genome_rate = 0.99;
            }
            else if heat == 0.1 && gene_rate == 0.99{
                gene_rate = 0.5;
                genome_rate = 0.5;
            }
            else if heat == 0.1 && gene_rate == 0.5{
                heat = 0.01;
                gene_rate = 0.99;
                genome_rate = 0.99;
            }
            else if heat == 0.01 && gene_rate == 0.99{
                gene_rate = 0.5;
                genome_rate = 0.5;
            }
            else{
                heat = 0.3;
                gene_rate = 0.99;
                genome_rate = 0.99;
            }

        }

        let elapsed = now.elapsed();

        println!(
            "Best: {:.3} d {:.3} d% {:.3} (heat {}; genome rate {}; dna rate {}) (time {:.3?})",
            best,
            prev_best - best,
            (prev_best - best) / prev_best,
            heat,
            genome_rate,
            gene_rate,
            elapsed
        );
        prev_best = best;
    }
}

pub fn small_evolutionary_training() {
    let (time_trace, full_syn_g, full_gap_g, _full_syn_e, sensory_indices) = read_data();

    let factory = Factory::new(&full_syn_g, &full_gap_g);
    let specification = factory.get_specification();

    let mut world = World::new();

    let file = File::open("results/small_evolution_latest_population.json").unwrap();
    let buffer = BufReader::new(file);
    let mut population: Vec<SmallGenome> = serde_json::from_reader(buffer).unwrap();

    for item in population.iter() {
        println!(
            "gap {} syn {} ex {} in {} leak g {} e {}",
            item.gap_g, item.syn_g, item.syn_e_ex, item.syn_e_in, item.leak_g, item.leak_e
        );
    }

    let mut heat = 1f64;

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let mut prev_best: f64 = 100000.0;

    for i in 0..10000 {
        if i == 5 {
            heat = 0.01;
        }

        let results: Vec<f64> = population
            .par_iter_mut()
            .map(|genome| factory.build_with_calc_gates(genome.expand(&specification)))
            .map(|mut model| {
                model
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
                    .error
            })
            .collect();

        let selection: Vec<(&SmallGenome, f64)> = population
            .iter()
            .zip(results)
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .take(10)
            .collect();

        let best = selection.first().unwrap().1;
        println!(
            "Best: {:.3} d {:.3} d% {:.3} (heat {})",
            best,
            prev_best - best,
            (prev_best - best) / prev_best,
            heat
        );
        prev_best = best;

        population = selection.iter().map(|x| x.0.clone()).collect();

        let file = File::create("results/small_evolution_latest_population.json").unwrap();
        let buffer = BufWriter::new(file);
        serde_json::to_writer(buffer, &population).unwrap();

        world.small_crossover(&mut population);

        population.push(population.first().unwrap().clone());

        world.small_mutate(&mut population, 0.99, 0.99, heat);
    }
}

pub fn gate_calculation() {
    let (time_trace, full_syn_g, full_gap_g, full_syn_e, sensory_indices) = read_data();

    let file = File::open("final_results/second_round_full_no_calc_gates_evolution_latest_population.json").unwrap();
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

    let genome = genomes.first().unwrap();

    let multiplier = 15.0;
    let adjust = -10.0;

    let mut model = factory.build(genome.clone());

    let voltage: Vec<f64> = (0..specification.model_len).map(|_| 0.0).collect();
    let gates: Vec<f64> = (0..specification.model_len).map(|_| 0.1).collect();

    let result = model.recorded_run_sensory(
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

    let file = File::create("results/first_round_full_run.json").unwrap();
    let buffer = BufWriter::new(file);
    serde_json::to_writer(buffer, &result.volt_record).unwrap();
}
