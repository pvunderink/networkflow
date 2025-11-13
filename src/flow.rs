use std::ops::{Add, Sub};

use num::Zero;

pub trait Flow: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Zero {}

macro_rules! impl_flow {
    ($($t:ty),*) => {
        $(
            impl Flow for $t {}
        )*
    };
}

impl_flow!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);
