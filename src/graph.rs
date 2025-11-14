use std::collections::HashMap;

use try_partialord::TryMinMax;

use crate::{EdgeHandle, FlowEdge, FlowNode, FlowNodeType, NodeHandle, cost::Cost, flow::Flow};

struct FlowPath<F, C>
where
    F: Flow,
{
    edges: Vec<EdgeHandle>,
    capacity: F,
    #[allow(dead_code)]
    total_cost: C,
}

impl<F, C> FlowPath<F, C>
where
    F: Flow,
    C: Cost,
{
    fn new(edges: Vec<FlowEdge<F, C>>) -> Self {
        let capacity = edges
            .iter()
            .map(|edge| edge.residual_capacity())
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

pub struct FlowGraph<F, C>
where
    F: Flow,
    C: Cost,
{
    nodes: Vec<FlowNode>,
    edges: HashMap<(NodeHandle, NodeHandle), Vec<FlowEdge<F, C>>>,
    edge_count: usize, // count of regular edges (i.e. excluding residual edges), used as edge_id
}

impl<F, C> FlowGraph<F, C>
where
    F: Flow,
    C: Cost,
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
        capacity: F,
        cost: C,
    ) -> EdgeHandle {
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
            capacity: F::zero(), // Initially zero capacity on the backward edge
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

    fn get_edges(&self, from: NodeHandle, to: NodeHandle) -> impl Iterator<Item = &FlowEdge<F, C>> {
        self.edges
            .get(&(from, to))
            .map(|edges| edges.iter())
            .into_iter()
            .flatten()
    }

    fn edges(&self) -> impl Iterator<Item = (NodeHandle, NodeHandle, &FlowEdge<F, C>)> {
        self.edges
            .iter()
            .flat_map(|((u, v), edges)| edges.iter().map(|e| (*u, *v, e)))
    }

    fn find_smallest_cost_path_with_capacity(
        &self,
        from: NodeHandle,
        to: NodeHandle,
    ) -> Option<FlowPath<F, C>> {
        let mut distances = vec![None; self.nodes.len()];
        let mut predecessors = vec![None; self.nodes.len()];
        distances[from.index] = Some(C::zero());

        // Relax edges |V| times (an extra iteration to detect negative cycles)
        for i in 0..self.nodes.len() {
            let mut updated = false;

            for (u, v, edge) in self.edges() {
                if edge.has_residual_capacity()
                    && let Some(distance) = distances[u.index]
                    && (distances[v.index].is_none()
                        || distance + edge.cost < distances[v.index].unwrap())
                {
                    if i == self.nodes.len() - 1 {
                        return None; // Negative cycle detected
                    }

                    distances[v.index] = Some(distance + edge.cost);
                    predecessors[v.index] = Some(u.index);
                    updated = true;
                }
            }

            if !updated {
                break;
            }
        }

        // Reconstruct path
        if distances[to.index].is_none() {
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

    fn get_edge_mut(&mut self, edge: EdgeHandle) -> Option<&mut FlowEdge<F, C>> {
        self.edges
            .get_mut(&(edge.from, edge.to))
            .and_then(|edges| edges.iter_mut().find(|e| e.edge_id == edge.edge_id))
    }

    fn augment_flow_along_path(&mut self, path: &FlowPath<F, C>) {
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
    /// The shortest paths are found using the Bellman-Ford algorithm to handle negative costs.
    pub fn maximize_flow(&mut self) {
        while let Some(path) =
            self.find_smallest_cost_path_with_capacity(self.source(), self.sink())
        {
            self.augment_flow_along_path(&path);
        }
    }

    pub fn get_flow(&self, edge: EdgeHandle) -> F {
        // The flow along an edge is equal to the capacity of the reverse edge
        let edge_entry = self
            .get_edges(edge.to, edge.from)
            .find(|e| e.edge_id == edge.edge_id)
            .expect("Edge not found");
        edge_entry.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph: FlowGraph<usize, f64> = FlowGraph::new();
        assert_eq!(graph.nodes.len(), 2); // Source and Sink
    }

    #[test]
    fn test_add_node_and_edge() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();
        let node_a = graph.add_node();
        let node_b = graph.add_node();
        graph.add_edge(node_a, node_b, 10, 1.0);
    }

    #[test]
    fn test_bellman_ford() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();
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
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();
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
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();
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
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();
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
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

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
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

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

    #[test]
    fn test_negative_cost_edge_flow() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, -2.0); // Negative cost
        let a_to_t = graph.add_edge(node_a, graph.sink(), 10, 1.0);

        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 10);
        assert_eq!(graph.get_flow(a_to_t), 10);
    }

    #[test]
    fn test_complex_network_with_multiple_paths() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();
        let node_b = graph.add_node();
        let node_c = graph.add_node();
        let node_d = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 5, 2.0);
        let s_to_b = graph.add_edge(graph.source(), node_b, 8, 1.0);
        let a_to_c = graph.add_edge(node_a, node_c, 4, 1.0);
        let b_to_c = graph.add_edge(node_b, node_c, 3, 3.0);
        let b_to_d = graph.add_edge(node_b, node_d, 6, 1.5);
        let c_to_t = graph.add_edge(node_c, graph.sink(), 7, 0.5);
        let d_to_t = graph.add_edge(node_d, graph.sink(), 9, 1.0);

        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 4);
        assert_eq!(graph.get_flow(s_to_b), 8);
        assert_eq!(graph.get_flow(a_to_c), 4);
        assert_eq!(graph.get_flow(b_to_c), 2);
        assert_eq!(graph.get_flow(b_to_d), 6);
        assert_eq!(graph.get_flow(c_to_t), 6);
        assert_eq!(graph.get_flow(d_to_t), 6);
    }

    #[test]
    fn test_zero_flow_due_to_high_cost() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1000.0); // Very high cost
        let a_to_t = graph.add_edge(node_a, graph.sink(), 10, 1.0);

        graph.maximize_flow();

        // Flow should still be maximized regardless of cost
        assert_eq!(graph.get_flow(s_to_a), 10);
        assert_eq!(graph.get_flow(a_to_t), 10);
    }

    #[test]
    fn test_bottleneck_with_cost_preference() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();
        let node_b = graph.add_node();
        let node_c = graph.add_node();

        // Two paths with different costs, but one has a bottleneck
        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1.0);
        let s_to_b = graph.add_edge(graph.source(), node_b, 10, 2.0);
        let a_to_c = graph.add_edge(node_a, node_c, 3, 1.0); // Bottleneck
        let b_to_c = graph.add_edge(node_b, node_c, 8, 1.0);
        let c_to_t = graph.add_edge(node_c, graph.sink(), 15, 1.0);

        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 3); // Limited by bottleneck
        assert_eq!(graph.get_flow(s_to_b), 8); // Uses remaining capacity
        assert_eq!(graph.get_flow(a_to_c), 3);
        assert_eq!(graph.get_flow(b_to_c), 8);
        assert_eq!(graph.get_flow(c_to_t), 11);
    }

    #[test]
    fn test_single_edge_maximum_flow() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let edge = graph.add_edge(graph.source(), graph.sink(), 7, 3.5);

        graph.maximize_flow();

        assert_eq!(graph.get_flow(edge), 7);
    }

    #[test]
    fn test_disconnected_nodes() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();
        let _node_b = graph.add_node(); // Disconnected node

        let s_to_a = graph.add_edge(graph.source(), node_a, 5, 1.0);
        let a_to_t = graph.add_edge(node_a, graph.sink(), 5, 1.0);

        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 5);
        assert_eq!(graph.get_flow(a_to_t), 5);
    }

    #[test]
    fn test_parallel_edges_different_costs() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1.0);
        let a_to_t_1 = graph.add_edge(node_a, graph.sink(), 4, 0.5); // Cheaper
        let a_to_t_2 = graph.add_edge(node_a, graph.sink(), 4, 2.0); // More expensive
        let a_to_t_3 = graph.add_edge(node_a, graph.sink(), 4, 1.0); // Middle cost

        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 10);
        assert_eq!(graph.get_flow(a_to_t_1), 4); // Should use cheapest first
        assert_eq!(graph.get_flow(a_to_t_3), 4); // Then middle cost
        assert_eq!(graph.get_flow(a_to_t_2), 2); // Finally most expensive
    }

    #[test]
    fn test_avoid_cycle_with_positive_cost() {
        let mut graph: FlowGraph<usize, f64> = FlowGraph::new();

        let node_a = graph.add_node();
        let node_b = graph.add_node();
        let node_c = graph.add_node();

        let s_to_a = graph.add_edge(graph.source(), node_a, 10, 1.0);
        let a_to_b = graph.add_edge(node_a, node_b, 10, 1.0);
        let b_to_c = graph.add_edge(node_b, node_c, 10, 1.0);
        let c_to_a = graph.add_edge(node_c, node_a, 5, 2.0); // Creates cycle
        let b_to_t = graph.add_edge(node_b, graph.sink(), 8, 1.0);

        graph.maximize_flow();

        assert_eq!(graph.get_flow(s_to_a), 8);
        assert_eq!(graph.get_flow(a_to_b), 8);
        assert_eq!(graph.get_flow(b_to_c), 0); // Should not use cycle
        assert_eq!(graph.get_flow(c_to_a), 0);
        assert_eq!(graph.get_flow(b_to_t), 8);
    }
}
