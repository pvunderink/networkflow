# networkflow

A Rust library for network flow algorithms.

## Features

- Minimum cost maximum flow calculation (based on Ford-Fulkerson with Bellman-Ford for shortest paths)

## Planned features

- An optimized solution for bipartite graphs
- An implementation of the Network Simplex algorithm 

## Quick Start

### Example

```rust
use networkflow::FlowGraph;

// Create a new flow graph with f64 as flow type and cost type
let mut graph: FlowGraph<f64, f64> = FlowGraph::new();
let node_a = graph.add_node();
let node_b = graph.add_node();
let node_c = graph.add_node();

let source_to_a = graph.add_edge(graph.source(), node_a, 10.0, 10.0); // Edge with high cost and high capacity
let source_to_b = graph.add_edge(graph.source(), node_b, 5.0, 1.0); // Edge with lower cost and lower capacity
graph.add_edge(node_a, node_c, f64::INFINITY, 1.0); // Infinite capacity edge with low cost
graph.add_edge(node_b, node_c, f64::INFINITY, 1.0); // Infinite capacity edge with low cost
let c_to_sink = graph.add_edge(node_c, graph.sink(), 10.0, 1.0); // Bottleneck edge to sink

// Maximize flow
graph.maximize_flow();

// Inspect flow along edges
assert_eq!(graph.get_flow(source_to_a), 5.0);
assert_eq!(graph.get_flow(source_to_b), 5.0);
assert_eq!(graph.get_flow(c_to_sink), 10.0);
```

## License

This project is available under the MIT license.