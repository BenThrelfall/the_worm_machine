use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Frame {
    pub time: f64,
    pub data: Vec<f64>,
}