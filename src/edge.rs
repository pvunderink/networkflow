use crate::{Cost, Flow, NodeHandle};

#[derive(Clone, Debug)]
pub(crate) struct FlowEdge<F, C>
where
    F: Flow,
    C: Cost,
{
    pub(crate) from: NodeHandle,
    pub(crate) to: NodeHandle,
    pub(crate) capacity: F,
    pub(crate) cost: C,
    pub(crate) edge_id: usize,
}

impl<F, C> FlowEdge<F, C>
where
    F: Flow,
    C: Cost,
{
    pub(crate) fn residual_capacity(&self) -> F {
        self.capacity
    }

    pub(crate) fn has_residual_capacity(&self) -> bool {
        self.capacity > F::zero()
    }

    pub(crate) fn handle(&self) -> EdgeHandle {
        EdgeHandle {
            from: self.from,
            to: self.to,
            edge_id: self.edge_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeHandle {
    pub(crate) from: NodeHandle,
    pub(crate) to: NodeHandle,
    pub(crate) edge_id: usize,
}

impl EdgeHandle {
    pub fn reverse(&self) -> EdgeHandle {
        EdgeHandle {
            from: self.to,
            to: self.from,
            edge_id: self.edge_id,
        }
    }
}
