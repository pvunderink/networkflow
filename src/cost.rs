use std::{
    iter::Sum,
    ops::{Add, Neg, Sub},
};

use num::Zero;

pub trait Cost:
    Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Sum + Neg<Output = Self> + Zero
{
}

macro_rules! impl_cost {
    ($($t:ty),*) => {
        $(
            impl Cost for $t {}
        )*
    };
}

impl_cost!(i8, i16, i32, i64, i128, isize, f32, f64);
