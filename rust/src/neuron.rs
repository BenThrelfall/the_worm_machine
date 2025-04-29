use itertools::izip;

use crate::data::{self, Frame};

const SYN_RISE: f64 = 1.0;
const SYN_DECAY: f64 = 2.0;

pub struct Network {
    pub syn_co: Vec<f64>,
    pub syn_int: Vec<f64>,
    pub gap_co: Vec<f64>,
    pub gap_int: Vec<f64>,
    pub leak_int: Vec<f64>,

    pub syn_indices: Vec<Vec<usize>>,
    pub syn_g: Vec<Vec<f64>>,
    pub syn_e: Vec<Vec<f64>>,
    pub leak_g: Vec<f64>,
    pub leak_e: Vec<f64>,
    pub gate_beta: Vec<f64>,
    pub gate_adjust: Vec<f64>,
    pub gap_indices: Vec<Vec<usize>>,
    pub gap_g: Vec<Vec<f64>>,
}

impl Network {
    pub fn new(
        full_syn_g: Vec<Vec<f64>>,
        full_syn_e: Vec<Vec<f64>>,
        gate_beta: Vec<f64>,
        gate_adjust: Vec<f64>,
        leak_g: Vec<f64>,
        leak_e: Vec<f64>,
        full_gap_g: Vec<Vec<f64>>,
    ) -> Network {
        let mut syn_g = Vec::new();
        let mut syn_e = Vec::new();
        let mut syn_indices = Vec::new();

        for (i, line) in full_syn_g.iter().enumerate() {
            syn_g.push(Vec::new());
            syn_e.push(Vec::new());
            syn_indices.push(Vec::new());

            for (j, val) in line.iter().enumerate() {
                if *val != 0f64 {
                    syn_g[i].push(*val);
                    syn_e[i].push(full_syn_e[i][j]);
                    syn_indices[i].push(j);
                }
            }
        }

        let mut gap_g = Vec::new();
        let mut gap_indices = Vec::new();

        for (i, line) in full_gap_g.iter().enumerate() {
            gap_g.push(Vec::new());
            gap_indices.push(Vec::new());

            for (j, val) in line.iter().enumerate() {
                if *val != 0f64 {
                    gap_g[i].push(*val);
                    gap_indices[i].push(j);
                }
            }
        }

        let syn_co: Vec<f64> = leak_g.clone();
        let syn_int: Vec<f64> = leak_g.clone();
        let gap_co: Vec<f64> = gap_g.iter().map(|line| line.iter().sum()).collect();
        let gap_int: Vec<f64> = leak_g.clone();
        let leak_int: Vec<f64> = leak_g
            .iter()
            .zip(leak_e.iter())
            .map(|(g, e)| g * e)
            .collect();

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

    pub fn run(
        &mut self,
        mut voltage: Vec<f64>,
        mut gates: Vec<f64>,
        timestep: f64,
        runtime: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut time = 0f64;

        while time < runtime {
            (voltage, gates) = self.step(voltage, gates, timestep);
            time += timestep;
        }

        return (voltage, gates);
    }

    pub fn recorded_run(
        &mut self,
        mut voltage: Vec<f64>,
        mut gates: Vec<f64>,
        timestep: f64,
        runtime: f64,
        stride: i32,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut time = 0f64;

        let mut volt_record = Vec::new();
        let mut gate_record = Vec::new();

        let mut frame_count = 0;

        while time < runtime {
            (voltage, gates) = self.step(voltage, gates, timestep);

            if frame_count % stride == 0 {
                volt_record.push(voltage.clone());
                gate_record.push(gates.clone());
            }
            frame_count += 1;

            time += timestep;
        }

        return (voltage, gates, volt_record, gate_record);
    }

    pub fn run_clamped(
        &mut self,
        mut voltage: Vec<f64>,
        mut gates: Vec<f64>,
        clamps: Vec<bool>,
        timestep: f64,
        runtime: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut time = 0f64;

        while time < runtime {
            let (new_voltage, new_gates) = self.step(voltage.clone(), gates, timestep);

            gates = new_gates;

            for i in 0..voltage.len() {
                if !clamps[i] {
                    voltage[i] = new_voltage[i];
                }
            }

            time += timestep;
        }

        let (_, _, internal_input) = self.recorded_step(voltage.clone(), gates.clone(), timestep);

        return (voltage, gates, internal_input);
    }

    pub fn recorded_run_sensory(
        &mut self,
        mut voltage: Vec<f64>,
        mut gates: Vec<f64>,
        timestep: f64,
        runtime: f64,
        trace: &Vec<Frame>,
        mul: f64,
        adjust: f64,
        indices: &Vec<usize>,
        stride: i32,
    ) -> SensoryRunResult {
        let mut time = 0f64;

        let mut volt_record = Vec::new();
        let mut gate_record = Vec::new();

        let mut frame_count = 0;

        let mut frame_a;
        let mut frame_b;
        let mut check_frame;
        let mut diff;
        let mut dist;

        let mut current_frame = 0;

        let mut processed_trace = trace.clone();
        data::preprocess_frames(&mut processed_trace, mul, adjust);

        current_frame += 1;

        frame_b = &processed_trace[current_frame];
        frame_a = &processed_trace[current_frame - 1];

        diff = frame_b.time - frame_a.time;

        let mut error = 0.0;

        while time < runtime {
            while processed_trace[current_frame].time <= time {
                current_frame += 1;

                frame_b = &processed_trace[current_frame];
                frame_a = &processed_trace[current_frame - 1];

                check_frame = &trace[current_frame - 1];

                diff = frame_b.time - frame_a.time;

                error += voltage
                    .iter()
                    .enumerate()
                    .zip(check_frame.data.iter())
                    .filter(|((i, _volt), _point)| indices.contains(i) == false)
                    .map(|((_, volt), point)| (((volt - adjust) / mul) - point).powf(2.0))
                    .sum::<f64>();
            }

            dist = (time - frame_a.time) / diff;

            for index in indices {
                if *index < frame_a.data.len() {
                    let lerp = frame_a.data[*index] * (1.0 - dist) + frame_b.data[*index] * dist;

                    //No Lerp
                    //let lerp = frame_a.data[*index];

                    // Replacing previous voltage
                    voltage[*index] = lerp;

                    //Input to voltage
                    //voltage[*index] += lerp * timestep;
                }
            }

            if frame_count % stride == 0 {
                volt_record.push(voltage.clone());
                gate_record.push(gates.clone());
            }

            (voltage, gates) = self.step(voltage, gates, timestep);

            frame_count += 1;

            time += timestep;
        }

        SensoryRunResult {
            voltage,
            gates,
            volt_record,
            gate_record,
            error,
        }
    }

    pub fn extra_recorded_run_sensory(
        &mut self,
        mut voltage: Vec<f64>,
        mut gates: Vec<f64>,
        timestep: f64,
        runtime: f64,
        trace: &Vec<Frame>,
        mul: f64,
        adjust: f64,
        indices: &Vec<usize>,
        stride: i32,
    ) -> ExtraSensoryRunResult {
        let mut time = 0f64;

        let mut volt_record = Vec::new();
        let mut gate_record = Vec::new();

        let mut frame_count = 0;

        let mut neuron_evals: Vec<f64> = (0..voltage.len()).map(|_| 0.0).collect();

        let mut frame_a;
        let mut frame_b;
        let mut check_frame;
        let mut diff;
        let mut dist;

        let mut evals_performed = 0;
        let mut neuron_wise_evals = 0;

        let mut current_frame = 0;

        let mut processed_trace = trace.clone();
        data::preprocess_frames(&mut processed_trace, mul, adjust);

        current_frame += 1;

        frame_b = &processed_trace[current_frame];
        frame_a = &processed_trace[current_frame - 1];

        diff = frame_b.time - frame_a.time;

        let mut error = 0.0;
        let mut proc_error = 0.0;

        let mut abs_error = 0.0;
        let mut proc_abs_error = 0.0;

        while time < runtime {
            while processed_trace[current_frame].time <= time {
                current_frame += 1;

                frame_b = &processed_trace[current_frame];
                frame_a = &processed_trace[current_frame - 1];

                check_frame = &trace[current_frame - 1];

                diff = frame_b.time - frame_a.time;

                evals_performed += 1;

                let loss = voltage.iter().enumerate().zip(check_frame.data.iter()).map(
                    |((i, volt), point)| {
                        if indices.contains(&i) {
                            NeuronError::new()
                        } else {
                            NeuronError{
                                mse: (((volt - adjust) / mul) - point).powf(2.0),
                                mabse: (((volt - adjust) / mul) - point).abs(),
                                proc_mse: (volt - (point * mul + adjust)).powf(2.0),
                                proc_mabse: (volt - (point * mul + adjust)).abs(),
                            }
                        }
                    },
                );

                for (i, item) in loss.enumerate() {
                    if item.mse != 0.0{
                        neuron_wise_evals += 1;
                    } 
                    neuron_evals[i] += item.proc_mabse;
                    error += item.mse;
                    proc_error += item.proc_mse;
                    abs_error += item.mabse;
                    proc_abs_error += item.proc_mabse;
                }
            }

            dist = (time - frame_a.time) / diff;

            for index in indices {
                if *index < frame_a.data.len() {
                    let lerp = frame_a.data[*index] * (1.0 - dist) + frame_b.data[*index] * dist;

                    //No Lerp
                    //let lerp = frame_a.data[*index];

                    // Replacing previous voltage
                    voltage[*index] = lerp;

                    //Input to voltage
                    //voltage[*index] += lerp * timestep;
                }
            }

            if frame_count % stride == 0 {
                volt_record.push(voltage.clone());
                gate_record.push(gates.clone());
            }

            (voltage, gates) = self.step(voltage, gates, timestep);

            frame_count += 1;

            time += timestep;
        }

        ExtraSensoryRunResult {
            voltage,
            gates,
            volt_record,
            gate_record,
            neuron_evals,
            error,
            evals_performed,
            raw_mse: error / neuron_wise_evals as f64,
            raw_mabse: abs_error / neuron_wise_evals as f64,
            proc_mse: proc_error / neuron_wise_evals as f64,
            proc_mabse: proc_abs_error / neuron_wise_evals as f64,
        }
    }

    fn step(&mut self, voltage: Vec<f64>, gates: Vec<f64>, timestep: f64) -> (Vec<f64>, Vec<f64>) {
        for (i, line) in self.syn_g.iter().enumerate() {
            let mut co_sum = 0f64;
            let mut int_sum = 0f64;

            for (j, val) in line.iter().enumerate() {
                let co = val * gates[self.syn_indices[i][j]];
                let int = co * self.syn_e[i][j];

                co_sum += co;
                int_sum += int;
            }

            self.syn_co[i] = co_sum;
            self.syn_int[i] = int_sum;
        }

        for (i, line) in self.gap_g.iter().enumerate() {
            self.gap_int[i] = line
                .iter()
                .zip(&self.gap_indices[i])
                .map(|(g, index)| g * voltage[*index])
                .sum()
        }

        let v_inf = izip!(
            &self.syn_co,
            &self.syn_int,
            &self.gap_co,
            &self.gap_int,
            &self.leak_g,
            &self.leak_int
        )
        .map(|(s_co, s_int, g_co, g_int, l_co, l_int)| {
            (l_int + s_int + g_int) / (l_co + s_co + g_co)
        });

        let leak_current = izip!(&voltage, &self.leak_g, &self.leak_e).map(|(v, g, e)| g * (v - e));
        let syn_current = izip!(&voltage, &self.syn_co, &self.syn_int).map(|(v, c, i)| c * v - i);
        let gap_current = izip!(&voltage, &self.gap_co, &self.gap_int).map(|(v, c, i)| c * v - i);

        let delta_voltage =
            izip!(leak_current, syn_current, gap_current).map(|(l, s, g)| (-l - s - g) * timestep);

        let delta_gates = izip!(&gates, &voltage, &self.gate_beta, &self.gate_adjust)
            .map(|(g, v, b, a)| Self::delta_s(*g, *v, *b, *a) * timestep);

        let new_gates: Vec<f64> = izip!(&gates, delta_gates).map(|(g, dg)| g + dg).collect();

        if voltage[0].abs() > 10000.0 {
            //panic!("Voltage is too large");
        }

        let new_voltage = izip!(&voltage, delta_voltage, v_inf)
            .map(|(volt, del, inf)| volt + del.clamp(-(volt - inf).abs(), (volt - inf).abs()))
            .collect();

        return (new_voltage, new_gates);
    }

    fn recorded_step(
        &mut self,
        voltage: Vec<f64>,
        gates: Vec<f64>,
        timestep: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        for (i, line) in self.syn_g.iter().enumerate() {
            let mut co_sum = 0f64;
            let mut int_sum = 0f64;

            for (j, val) in line.iter().enumerate() {
                let co = val * gates[self.syn_indices[i][j]];
                let int = co * self.syn_e[i][j];

                co_sum += co;
                int_sum += int;
            }

            self.syn_co[i] = co_sum;
            self.syn_int[i] = int_sum;
        }

        for (i, line) in self.gap_g.iter().enumerate() {
            self.gap_int[i] = line
                .iter()
                .zip(&self.gap_indices[i])
                .map(|(g, index)| g * voltage[*index])
                .sum()
        }

        let v_inf = izip!(
            &self.syn_co,
            &self.syn_int,
            &self.gap_co,
            &self.gap_int,
            &self.leak_g,
            &self.leak_int
        )
        .map(|(s_co, s_int, g_co, g_int, l_co, l_int)| {
            (l_int + s_int + g_int) / (l_co + s_co + g_co)
        });

        let leak_current = izip!(&voltage, &self.leak_g, &self.leak_e).map(|(v, g, e)| g * (v - e));
        let syn_current = izip!(&voltage, &self.syn_co, &self.syn_int).map(|(v, c, i)| c * v - i);
        let gap_current = izip!(&voltage, &self.gap_co, &self.gap_int).map(|(v, c, i)| c * v - i);

        let internal_input: Vec<f64> = izip!(syn_current.clone(), gap_current.clone())
            .map(|(s, g)| s + g)
            .collect();

        let delta_voltage =
            izip!(leak_current, syn_current, gap_current).map(|(l, s, g)| (-l - s - g) * timestep);

        let delta_gates = izip!(&gates, &voltage, &self.gate_beta, &self.gate_adjust)
            .map(|(g, v, b, a)| Self::delta_s(*g, *v, *b, *a) * timestep);

        let new_gates: Vec<f64> = izip!(&gates, delta_gates).map(|(g, dg)| g + dg).collect();

        if voltage[0].abs() > 10000.0 {
            panic!("Voltage is too large");
        }

        let new_voltage = izip!(&voltage, delta_voltage, v_inf)
            .map(|(volt, del, inf)| volt + del.clamp(-(volt - inf).abs(), (volt - inf).abs()))
            .collect();

        return (new_voltage, new_gates, internal_input);
    }

    fn delta_s(gate_value: f64, voltage: f64, beta: f64, adjust: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-beta * (voltage - adjust)).exp());
        return SYN_RISE * sigmoid * (1.0 - gate_value) - SYN_DECAY * gate_value;
    }

    fn direct_s(voltage: f64, beta: f64, adjust: f64) -> f64 {
        let sig = 1.0 / (1.0 + (-beta * (voltage - adjust)).exp());
        return (1.0 * sig) / (1.0 * sig + 5.0);
    }
}

pub struct SensoryRunResult {
    pub voltage: Vec<f64>,
    pub gates: Vec<f64>,
    pub volt_record: Vec<Vec<f64>>,
    pub gate_record: Vec<Vec<f64>>,
    pub error: f64,
}

pub struct ExtraSensoryRunResult {
    pub voltage: Vec<f64>,
    pub gates: Vec<f64>,
    pub volt_record: Vec<Vec<f64>>,
    pub gate_record: Vec<Vec<f64>>,
    pub neuron_evals: Vec<f64>,
    pub error: f64,
    pub evals_performed: u32,
    pub raw_mse: f64,
    pub raw_mabse: f64,
    pub proc_mse: f64,
    pub proc_mabse: f64,
}

struct NeuronError{
    pub mse: f64,
    pub mabse: f64,
    pub proc_mse: f64,
    pub proc_mabse: f64,
}

impl NeuronError{
    pub fn new() -> Self{
        NeuronError{
            mse: 0.0,
            mabse: 0.0,
            proc_mse: 0.0,
            proc_mabse: 0.0,
        }
    }
}