use std::{
    collections::HashMap,
    fmt::Debug,
    iter::Sum,
    ops::{Add, Div, Sub},
};

use num::{One, Zero};
use try_partialord::TryMinMax;

pub enum FlowNodeType {
    Source,
    Sink,
    Inner,
}

struct FlowNode {
    node_type: FlowNodeType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    index: usize,
}

#[derive(Clone, Debug)]
struct FlowEdge<C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero,
{
    from: NodeHandle,
    to: NodeHandle,
    capacity: C,
    cost: f64,
    edge_id: usize,
}

impl<C> FlowEdge<C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero,
{
    fn residual_capacity(&self) -> C {
        self.capacity
    }

    fn has_residual_capacity(&self) -> bool {
        self.capacity > C::zero()
    }

    fn handle(&self) -> FlowEdgeHandle {
        FlowEdgeHandle {
            from: self.from,
            to: self.to,
            edge_id: self.edge_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlowEdgeHandle {
    from: NodeHandle,
    to: NodeHandle,
    edge_id: usize,
}

impl FlowEdgeHandle {
    fn reverse(&self) -> FlowEdgeHandle {
        FlowEdgeHandle {
            from: self.to,
            to: self.from,
            edge_id: self.edge_id,
        }
    }
}

struct FlowPath<C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero,
{
    edges: Vec<FlowEdgeHandle>,
    capacity: C,
    #[cfg(test)]
    total_cost: f64,
}

impl<C> FlowPath<C>
where
    C: Copy
        + PartialOrd
        + Add<Output = C>
        + Sub<Output = C>
        + Zero
        + One
        + Sum
        + Div<Output = C>
        + Debug,
{
    fn new(edges: Vec<FlowEdge<C>>) -> Self {
        let capacity = edges
            .iter()
            .map(|edge| edge.residual_capacity())
            .try_min()
            .expect("Invalid capacities on edges")
            .expect("Empty path");

        #[cfg(test)]
        let total_cost = edges.iter().map(|edge| edge.cost).sum();

        FlowPath {
            edges: edges.into_iter().map(|e| e.handle()).collect(),
            capacity,
            #[cfg(test)]
            total_cost,
        }
    }
}

pub struct FlowGraph<C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero + One + Sum + Div<Output = C>,
{
    nodes: Vec<FlowNode>,
    edges: HashMap<(NodeHandle, NodeHandle), Vec<FlowEdge<C>>>,
    edge_count: usize, // count of regular edges (i.e. excluding residual edges), used as edge_id
}

impl<C> FlowGraph<C>
where
    C: Copy
        + PartialOrd
        + Add<Output = C>
        + Sub<Output = C>
        + Zero
        + One
        + Sum
        + Div<Output = C>
        + Debug,
{
    pub fn new() -> Self {
        let mut graph = FlowGraph {
            nodes: Vec::new(),
            edges: HashMap::new(),
            edge_count: 0,
        };

        graph.add_node_raw(FlowNodeType::Source);
        graph.add_node_raw(FlowNodeType::Sink);

        graph
    }

    pub fn source(&self) -> NodeHandle {
        NodeHandle { index: 0 }
    }

    pub fn sink(&self) -> NodeHandle {
        NodeHandle { index: 1 }
    }

    fn add_node_raw(&mut self, node_type: FlowNodeType) -> NodeHandle {
        let index = self.nodes.len();
        self.nodes.push(FlowNode { node_type });
        NodeHandle { index }
    }

    pub fn add_node(&mut self) -> NodeHandle {
        self.add_node_raw(FlowNodeType::Inner)
    }

    pub fn add_edge(
        &mut self,
        from: NodeHandle,
        to: NodeHandle,
        capacity: C,
        cost: f64,
    ) -> FlowEdgeHandle {
        let edge_id = self.edge_count;
        self.edge_count += 1;

        let forward_edge = FlowEdge {
            from,
            to,
            capacity,
            cost,
            edge_id,
        };
        let handle = forward_edge.handle();

        let backward_edge = FlowEdge {
            from: to,
            to: from,
            capacity: C::zero(), // Initially zero capacity on the backward edge
            cost: -cost,
            edge_id,
        };

        if let Some(edges) = self.edges.get_mut(&(from, to)) {
            edges.push(forward_edge);
        } else {
            self.edges.insert((from, to), vec![forward_edge]);
        }

        if let Some(edges) = self.edges.get_mut(&(to, from)) {
            edges.push(backward_edge);
        } else {
            self.edges.insert((to, from), vec![backward_edge]);
        }

        handle
    }

    fn get_edges(&self, from: NodeHandle, to: NodeHandle) -> impl Iterator<Item = &FlowEdge<C>> {
        self.edges
            .get(&(from, to))
            .map(|edges| edges.iter())
            .into_iter()
            .flatten()
    }

    fn edges(&self) -> impl Iterator<Item = (NodeHandle, NodeHandle, &FlowEdge<C>)> {
        self.edges
            .iter()
            .flat_map(|((u, v), edges)| edges.iter().map(|e| (*u, *v, e)))
    }

    fn find_smallest_cost_path_with_capacity(
        &self,
        from: NodeHandle,
        to: NodeHandle,
    ) -> Option<FlowPath<C>> {
        let mut distances = vec![f64::INFINITY; self.nodes.len()];
        let mut predecessors = vec![None; self.nodes.len()];
        distances[from.index] = 0.0;

        // Relax edges |V| times (an extra iteration to detect negative cycles)
        for i in 0..self.nodes.len() {
            let mut updated = false;

            for (u, v, edge) in self.edges() {
                if edge.has_residual_capacity()
                    && distances[u.index] != f64::INFINITY
                    && distances[u.index] + edge.cost < distances[v.index]
                {
                    if i == self.nodes.len() - 1 {
                        return None; // Negative cycle detected
                    }

                    distances[v.index] = distances[u.index] + edge.cost;
                    predecessors[v.index] = Some(u.index);
                    updated = true;
                }
            }

            if !updated {
                break;
            }
        }

        // Reconstruct path
        if distances[to.index] == f64::INFINITY {
            return None; // No path exists
        }

        let mut path = Vec::new();
        let mut current = to.index;
        while let Some(pred) = predecessors[current] {
            // Find the cheapest edge with residual capacity between pred and current
            let cheapest_edge = self
                .get_edges(NodeHandle { index: pred }, NodeHandle { index: current })
                .filter(|edge| edge.has_residual_capacity())
                .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
                .expect("No edge found with residual capacity during path reconstruction");

            path.push(cheapest_edge.clone());
            current = pred;
        }
        path.reverse();

        Some(FlowPath::new(path))
    }

    fn get_edge_mut(&mut self, edge: FlowEdgeHandle) -> Option<&mut FlowEdge<C>> {
        self.edges
            .get_mut(&(edge.from, edge.to))
            .and_then(|edges| edges.iter_mut().find(|e| e.edge_id == edge.edge_id))
    }

    fn augment_flow_along_path(&mut self, path: &FlowPath<C>) {
        for edge in &path.edges {
            let edge_entry = self.get_edge_mut(*edge).expect("Edge not found");
            edge_entry.capacity = edge_entry.capacity - path.capacity;

            let backward_edge_entry = self
                .get_edge_mut(edge.reverse())
                .expect("Backward edge not found");

            backward_edge_entry.capacity = backward_edge_entry.capacity + path.capacity;
        }
    }

    /// Maximize flow from source to sink while minimizing cost, using Ford-Fulkerson with successive shortest paths.
    ///
    /// ### Termination
    /// This function always terminates if there are no shared capacity groups, or if using non-integer capacities.
    ///
    /// ### Non-termination
    /// This function might not terminate if using shared capacity groups with integer capacities.
    /// Particularly, if edges with shared capacity are along the same path and the shared capacity is not divisible by
    /// the number of edges.
    pub fn maximize_flow(&mut self) {
        while let Some(path) =
            self.find_smallest_cost_path_with_capacity(self.source(), self.sink())
        {
            self.augment_flow_along_path(&path);
        }
    }

    pub fn get_flow(&self, edge: FlowEdgeHandle) -> C {
        // The flow along an edge is equal to the capacity of the reverse edge
        let edge_entry = self
            .get_edges(edge.to, edge.from)
            .find(|e| e.edge_id == edge.edge_id)
            .expect("Edge not found");
        edge_entry.capacity
    }

    pub fn get_node_type(&self, handle: NodeHandle) -> &FlowNodeType {
        &self.nodes[handle.index].node_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph: FlowGraph<usize> = FlowGraph::new();
        assert_eq!(graph.nodes.len(), 2); // Source and Sink
    }

    #[test]
    fn test_add_node_and_edge() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();
        let node_a = graph.add_node();
        let node_b = graph.add_node();
        graph.add_edge(node_a, node_b, 10, 1.0);
    }

    #[test]
    fn test_bellman_ford() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();
        let node_a = graph.add_node();
        let node_b = graph.add_node();
        graph.add_edge(graph.source(), node_a, 10, 1.0);
        graph.add_edge(node_a, node_b, 5, 1.0);
        graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let path = graph.find_smallest_cost_path_with_capacity(graph.source(), graph.sink());
        assert!(path.is_some());
    }

    #[test]
    fn test_bellman_with_cheaper_path() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();
        let node_a = graph.add_node();
        let node_b = graph.add_node();
        graph.add_edge(graph.source(), node_a, 10, 5.0);
        graph.add_edge(node_a, node_b, 5, 10.0);
        graph.add_edge(node_a, node_b, 5, -1.0);
        graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let path = graph
            .find_smallest_cost_path_with_capacity(graph.source(), graph.sink())
            .expect("No path found");
        assert_eq!(path.total_cost, 5.0 + -1.0 + 1.0);
    }

    #[test]
    fn test_bellman_ford_with_negative_cycle() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();
        let node_a = graph.add_node();
        let node_b = graph.add_node();
        graph.add_edge(graph.source(), node_a, 10, 5.0);
        graph.add_edge(node_a, node_b, 5, 10.0);
        graph.add_edge(node_b, node_a, 5, -11.0);
        graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let path = graph.find_smallest_cost_path_with_capacity(graph.source(), graph.sink());
        assert!(path.is_none(), "Negative cycle should be detected");
    }

    #[test]
    fn test_bellman_ford_with_no_capacity() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();
        let node_a = graph.add_node();
        let node_b = graph.add_node();
        graph.add_edge(graph.source(), node_a, 10, 5.0);
        graph.add_edge(node_a, node_b, 0, 10.0);
        graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let path = graph.find_smallest_cost_path_with_capacity(graph.source(), graph.sink());
        assert!(
            path.is_none(),
            "No path should be found due to zero capacity"
        );
    }

    #[test]
    fn test_maximize_flow() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();

        let node_a = graph.add_node();
        let node_b = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1.0);
        let a_to_b = graph.add_edge(node_a, node_b, 5, 1.0);
        let b_to_t = graph.add_edge(node_b, graph.sink(), 10, 1.0);

        // act
        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 5);
        assert_eq!(graph.get_flow(a_to_b), 5);
        assert_eq!(graph.get_flow(b_to_t), 5);
    }

    #[test]
    fn test_maximize_flow_prefers_cheaper_path() {
        let mut graph: FlowGraph<usize> = FlowGraph::new();

        let node_a = graph.add_node();
        let node_b = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1.0);
        let a_to_b_1 = graph.add_edge(node_a, node_b, 5, 1.0);
        let a_to_b_2 = graph.add_edge(node_a, node_b, 5, -1.0);
        let b_to_t = graph.add_edge(node_b, graph.sink(), 6, 1.0);

        // act
        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 6);
        assert_eq!(graph.get_flow(a_to_b_1), 1);
        assert_eq!(graph.get_flow(a_to_b_2), 5);
        assert_eq!(graph.get_flow(b_to_t), 6);
    }
}
