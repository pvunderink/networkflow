use std::ops::{Add, Sub};

use num::Zero;

pub trait Flow: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Zero {}

impl<T> Flow for T where T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Zero {}
