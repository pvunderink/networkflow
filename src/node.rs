pub(crate) enum FlowNodeType {
    Source,
    Sink,
    Inner,
}

pub(crate) struct FlowNode {
    #[allow(dead_code)]
    pub(crate) node_type: FlowNodeType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    pub(crate) index: usize,
}
