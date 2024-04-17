use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Frame {
    pub time: f64,
    pub data: Vec<f64>,
}

pub fn preprocess_frames(time_trace: &mut Vec<Frame>, multiplier: f64, post_adjust: f64){
    for frame in time_trace.iter_mut() {
        for point in frame.data.iter_mut() {
            *point = multiplier * *point + post_adjust;
        }
    }
}

pub fn unprocess_frames(time_trace: &mut Vec<Frame>, multiplier: f64, post_adjust: f64){
    for frame in time_trace.iter_mut() {
        for point in frame.data.iter_mut() {
            *point = (*point - post_adjust) / multiplier;
        }
    }
}