use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    iter::Sum,
    ops::{Add, Div, Sub},
};

use num::{One, Zero};
use try_partialord::TryMinMax;

pub enum FlowNodeType<T> {
    Source,
    Sink,
    Inner(T),
}

struct FlowNode<T> {
    index: usize,
    node_type: FlowNodeType<T>,
}

impl<T> FlowNode<T> {
    fn handle(&self) -> NodeHandle {
        NodeHandle { index: self.index }
    }
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

pub struct FlowPath<C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero,
{
    edges: Vec<FlowEdgeHandle>,
    capacity: C,
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
    fn new<T>(graph: &FlowGraph<T, C>, edges: Vec<FlowEdge<C>>) -> Self {
        let minimum_group = graph
            .groups
            .iter()
            .filter_map(|group| {
                let group_capacity = group.capacity_increment_of_edges(graph, &edges)?;
                if group_capacity < C::zero() {
                    None
                } else {
                    Some((group, group_capacity))
                }
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let capacity = edges
            .iter()
            .map(|edge| {
                let edge_capacity = graph.get_edge_capacity(edge);
                if let Some((minimum_group, minimum_group_capacity)) = minimum_group {
                    if minimum_group.edges.contains(&edge.handle())
                        && minimum_group_capacity < edge_capacity
                    {
                        minimum_group_capacity
                    } else {
                        edge_capacity
                    }
                } else {
                    edge_capacity
                }
            })
            .try_min()
            .expect("Invalid capacities on edges")
            .expect("Empty path");

        let total_cost = edges.iter().map(|edge| edge.cost).sum();

        FlowPath {
            edges: edges.into_iter().map(|e| e.handle()).collect(),
            capacity,
            total_cost,
        }
    }
}

pub struct SharedCapacityGroup<C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero + Sum,
{
    capacity: C,
    edges: HashSet<FlowEdgeHandle>,
}

impl<C> SharedCapacityGroup<C>
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
    pub fn new(capacity: C, edges: &[FlowEdgeHandle]) -> Self {
        SharedCapacityGroup {
            capacity,
            edges: edges.iter().cloned().collect(),
        }
    }

    pub fn empty_with_capacity(capacity: C) -> Self {
        SharedCapacityGroup {
            capacity,
            edges: HashSet::new(),
        }
    }

    pub fn add_edge(&mut self, edge: FlowEdgeHandle) {
        self.edges.insert(edge);
    }

    /// Calculate the residual capacity available in this group.
    fn residual_capacity<T>(&self, graph: &FlowGraph<T, C>) -> C {
        let used_capacity: C = self.edges.iter().map(|edge| graph.get_flow(*edge)).sum();
        self.capacity - used_capacity
    }

    /// Find the capacity increment available for the given edges in this group.
    /// Returns None if none of the edges are part of this group.
    ///
    /// The increment is divided equally among the edges in the group that are part of the given edges.
    /// This does not work for integers if the capacity is not divisible by the number of edges.
    fn capacity_increment_of_edges<T>(
        &self,
        graph: &FlowGraph<T, C>,
        edges: &Vec<FlowEdge<C>>,
    ) -> Option<C> {
        let used_capacity: C = edges
            .iter()
            .filter(|edge| self.edges.contains(&edge.handle()))
            .map(|edge| graph.get_flow(edge.handle()))
            .sum();

        let count = edges
            .iter()
            .filter_map(|edge| self.edges.contains(&edge.handle()).then(|| C::one()))
            .sum();

        if count == C::zero() {
            return None;
        }

        Some((self.capacity - used_capacity) / count)
    }
}

pub struct FlowGraph<T, C>
where
    C: Copy + PartialOrd + Add<Output = C> + Sub<Output = C> + Zero + One + Sum + Div<Output = C>,
{
    nodes: Vec<FlowNode<T>>,
    edges: HashMap<(NodeHandle, NodeHandle), Vec<FlowEdge<C>>>,
    edge_count: usize, // count of regular edges (i.e. excluding residual edges), used as edge_id
    groups: Vec<SharedCapacityGroup<C>>,
}

impl<T, C> FlowGraph<T, C>
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
            groups: Vec::new(),
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

    fn add_node_raw(&mut self, node_type: FlowNodeType<T>) -> NodeHandle {
        let index = self.nodes.len();
        self.nodes.push(FlowNode { index, node_type });
        NodeHandle { index }
    }

    pub fn add_node(&mut self, node: T) -> NodeHandle {
        self.add_node_raw(FlowNodeType::Inner(node))
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

    pub fn add_shared_capacity_group(&mut self, capacity: C, edges: &[FlowEdgeHandle]) {
        let group = SharedCapacityGroup::new(capacity, edges);
        self.groups.push(group)
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

    fn get_edge_capacity_groups(
        &self,
        edge: FlowEdgeHandle,
    ) -> impl Iterator<Item = &SharedCapacityGroup<C>> {
        self.groups
            .iter()
            .filter(move |group| group.edges.contains(&edge))
    }

    fn get_edge_capacity(&self, edge: &FlowEdge<C>) -> C {
        // Check if the edge is part of any shared capacity group
        if let Some(minimum_group_capacity) = self
            .groups
            .iter()
            .filter(|group| group.edges.contains(&edge.handle()))
            .map(|group| group.residual_capacity(self))
            .try_min()
            .expect("Invalid shared capacity")
            && minimum_group_capacity < edge.residual_capacity()
        {
            minimum_group_capacity
        } else {
            edge.residual_capacity()
        }
    }

    fn has_residual_capacity(&self, edge: &FlowEdge<C>) -> bool {
        self.get_edge_capacity(edge) > C::zero()
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
                if self.has_residual_capacity(edge)
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
                .filter(|edge| self.has_residual_capacity(edge))
                .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
                .expect("No edge found with residual capacity during path reconstruction");

            path.push(cheapest_edge.clone());
            current = pred;
        }
        path.reverse();

        Some(FlowPath::new(self, path))
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

    pub fn get_node_type(&self, handle: NodeHandle) -> &FlowNodeType<T> {
        &self.nodes[handle.index].node_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph: FlowGraph<(), usize> = FlowGraph::new();
        assert_eq!(graph.nodes.len(), 2); // Source and Sink
    }

    #[test]
    fn test_add_node_and_edge() {
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();
        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
        graph.add_edge(node_a, node_b, 10, 1.0);
    }

    #[test]
    fn test_bellman_ford() {
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();
        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
        graph.add_edge(graph.source(), node_a, 10, 1.0);
        graph.add_edge(node_a, node_b, 5, 1.0);
        graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let path = graph.find_smallest_cost_path_with_capacity(graph.source(), graph.sink());
        assert!(path.is_some());
    }

    #[test]
    fn test_bellman_with_cheaper_path() {
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();
        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
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
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();
        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
        graph.add_edge(graph.source(), node_a, 10, 5.0);
        graph.add_edge(node_a, node_b, 5, 10.0);
        graph.add_edge(node_b, node_a, 5, -11.0);
        graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let path = graph.find_smallest_cost_path_with_capacity(graph.source(), graph.sink());
        assert!(path.is_none(), "Negative cycle should be detected");
    }

    #[test]
    fn test_bellman_ford_with_no_capacity() {
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();
        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
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
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();

        let node_a = graph.add_node(());
        let node_b = graph.add_node(());

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
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();

        let node_a = graph.add_node(());
        let node_b = graph.add_node(());

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

    #[test]
    fn test_maximize_flow_with_shared_capacity() {
        let mut graph: FlowGraph<(), usize> = FlowGraph::new();

        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
        let node_c = graph.add_node(());

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1.0);
        let s_to_c = graph.add_edge(graph.source(), node_c, 10, 1.0);
        let a_to_b = graph.add_edge(node_a, node_b, 10, 1.0);
        let b_to_t = graph.add_edge(node_b, graph.sink(), 10, 1.0);
        let c_to_t = graph.add_edge(node_c, graph.sink(), 10, 1.0);

        graph.add_shared_capacity_group(5, &[a_to_b, s_to_c]);

        // act
        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 0);
        assert_eq!(graph.get_flow(s_to_c), 5);
        assert_eq!(graph.get_flow(a_to_b), 0);
        assert_eq!(graph.get_flow(b_to_t), 0);
        assert_eq!(graph.get_flow(c_to_t), 5);
    }

    #[test]
    fn test_maximize_flow_with_shared_capacity_along_same_path() {
        let mut graph: FlowGraph<(), f64> = FlowGraph::new();

        let node_a = graph.add_node(());
        let node_b = graph.add_node(());
        let node_c = graph.add_node(());

        let s_to_a = graph.add_edge(graph.source(), node_a, 10., 1.0);
        let s_to_c = graph.add_edge(graph.source(), node_c, 10., 1.0);
        let a_to_b = graph.add_edge(node_a, node_b, 10., 1.0);
        let b_to_t = graph.add_edge(node_b, graph.sink(), 10., 1.0);
        let c_to_t = graph.add_edge(node_c, graph.sink(), 10., 1.0);

        graph.add_shared_capacity_group(5., &[s_to_a, a_to_b]);

        // act
        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 2.5);
        assert_eq!(graph.get_flow(s_to_c), 10.);
        assert_eq!(graph.get_flow(a_to_b), 2.5);
        assert_eq!(graph.get_flow(b_to_t), 2.5);
        assert_eq!(graph.get_flow(c_to_t), 10.);
    }
}
