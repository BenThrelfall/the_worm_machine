use crate::{neuron::Network, Frame};

const DEFAULT_VOLTAGE: f64 = 0.0;
const DEFAULT_GATE: f64 = 0.0;

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

        error += voltage.iter()
                        .zip(points)
                        .map(|(volt, point)| (volt - point).powf(2.0))
                        .sum::<f64>();
    }

    return error;
}
