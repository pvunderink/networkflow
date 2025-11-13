use core::f64;

use energy_flow::graph::FlowGraph;

const HIGHLY_PREFER: isize = 1;
const PREFER: isize = 10;
const NEUTRAL: isize = 100;
const AVOID: isize = 1000;

fn main() {
    let mut graph = FlowGraph::new();

    let solar = graph.add_node();
    let grid_source = graph.add_node();
    let grid_sink = graph.add_node();
    let battery_source = graph.add_node();
    let battery_sink = graph.add_node();
    let charger = graph.add_node();

    let solar_to_charger = graph.add_edge(solar, charger, f64::INFINITY, HIGHLY_PREFER);
    let solar_to_grid_sink = graph.add_edge(solar, grid_sink, f64::INFINITY, AVOID);
    let solar_to_battery_sink = graph.add_edge(solar, battery_sink, f64::INFINITY, PREFER);

    let grid_source_to_charger = graph.add_edge(grid_source, charger, f64::INFINITY, AVOID);
    let grid_source_to_battery_sink =
        graph.add_edge(grid_source, battery_sink, f64::INFINITY, NEUTRAL);

    let battery_source_to_charger = graph.add_edge(battery_source, charger, f64::INFINITY, PREFER);
    let battery_source_to_grid_sink =
        graph.add_edge(battery_source, grid_sink, f64::INFINITY, NEUTRAL);

    let from_solar = 40.0;
    let from_grid = 30.0;
    let from_battery = 10.0;

    let to_grid = 10.0;
    let to_battery = 50.0;
    let to_charger = 20.0;

    // Supply
    graph.add_edge(graph.source(), solar, from_solar, NEUTRAL);
    graph.add_edge(graph.source(), grid_source, from_grid, NEUTRAL);
    graph.add_edge(graph.source(), battery_source, from_battery, NEUTRAL);

    // Demand
    graph.add_edge(charger, graph.sink(), to_charger, NEUTRAL);
    graph.add_edge(battery_sink, graph.sink(), to_battery, NEUTRAL);
    graph.add_edge(grid_sink, graph.sink(), to_grid, NEUTRAL);

    graph.maximize_flow();

    println!("Solar -[{}]-> Charger", graph.get_flow(solar_to_charger));
    println!("Solar -[{}]-> Grid", graph.get_flow(solar_to_grid_sink));
    println!(
        "Solar -[{}]-> Battery",
        graph.get_flow(solar_to_battery_sink)
    );

    println!(
        "Grid -[{}]-> Charger",
        graph.get_flow(grid_source_to_charger)
    );
    println!(
        "Grid -[{}]-> Battery",
        graph.get_flow(grid_source_to_battery_sink)
    );

    println!(
        "Battery -[{}]-> Charger",
        graph.get_flow(battery_source_to_charger)
    );
    println!(
        "Battery -[{}]-> Grid",
        graph.get_flow(battery_source_to_grid_sink)
    );
}

// use mcmf::{Capacity, Cost, GraphBuilder, Vertex};
// fn main() {
//     let from_solar = 40;
//     let from_grid = 30;
//     let from_battery = 10;

//     let to_grid = 10;
//     let to_battery = 50;
//     let to_charger = 20;

//     // Build the energy flow graph using mcmf
//     let (total_cost, paths) = GraphBuilder::new()
//         // Supply edges from source (scaled to integers)
//         .add_edge(
//             Vertex::Source,
//             "solar",
//             Capacity(from_solar),
//             Cost(NEUTRAL as i32),
//         )
//         .add_edge(
//             Vertex::Source,
//             "grid_source",
//             Capacity(from_grid),
//             Cost(NEUTRAL as i32),
//         )
//         .add_edge(
//             Vertex::Source,
//             "battery_source",
//             Capacity(from_battery),
//             Cost(NEUTRAL as i32),
//         )
//         // Energy routing edges with preferences - use large capacity
//         .add_edge(
//             "solar",
//             "charger",
//             Capacity(10000),
//             Cost(HIGHLY_PREFER as i32),
//         )
//         .add_edge("solar", "grid_sink", Capacity(10000), Cost(AVOID as i32))
//         .add_edge(
//             "solar",
//             "battery_sink",
//             Capacity(10000),
//             Cost(PREFER as i32),
//         )
//         .add_edge(
//             "grid_source",
//             "charger",
//             Capacity(10000),
//             Cost(PREFER as i32),
//         )
//         .add_edge(
//             "grid_source",
//             "battery_sink",
//             Capacity(10000),
//             Cost(NEUTRAL as i32),
//         )
//         .add_edge(
//             "battery_source",
//             "charger",
//             Capacity(10000),
//             Cost(PREFER as i32),
//         )
//         .add_edge(
//             "battery_source",
//             "grid_sink",
//             Capacity(10000),
//             Cost(NEUTRAL as i32),
//         )
//         // Demand edges to sink (scaled to integers)
//         .add_edge(
//             "charger",
//             Vertex::Sink,
//             Capacity(to_charger),
//             Cost(NEUTRAL as i32),
//         )
//         .add_edge(
//             "battery_sink",
//             Vertex::Sink,
//             Capacity(to_battery),
//             Cost(NEUTRAL as i32),
//         )
//         .add_edge(
//             "grid_sink",
//             Vertex::Sink,
//             Capacity(to_grid),
//             Cost(NEUTRAL as i32),
//         )
//         .mcmf();

//     println!("Total cost: {}", total_cost);
//     println!("Number of flow paths: {}", paths.len());

//     // Analyze the flow paths to show energy routing
//     let mut solar_to_charger = 0;
//     let mut solar_to_grid = 0;
//     let mut solar_to_battery = 0;
//     let mut grid_to_charger = 0;
//     let mut grid_to_battery = 0;
//     let mut battery_to_charger = 0;
//     let mut battery_to_grid = 0;

//     for path in &paths {
//         let vertices = path.vertices();
//         let path_cost = path.cost();

//         println!("Path: {:?}, Cost: {}", vertices, path_cost);

//         // Each path in mcmf represents 1 unit of flow
//         let flow = path.amount();

//         // Match flow patterns based on the path vertices
//         if vertices.len() >= 4 {
//             match (vertices[1], vertices[2]) {
//                 (Vertex::Node("solar"), Vertex::Node("charger")) => solar_to_charger += flow,
//                 (Vertex::Node("solar"), Vertex::Node("grid_sink")) => solar_to_grid += flow,
//                 (Vertex::Node("solar"), Vertex::Node("battery_sink")) => solar_to_battery += flow,
//                 (Vertex::Node("grid_source"), Vertex::Node("charger")) => grid_to_charger += flow,
//                 (Vertex::Node("grid_source"), Vertex::Node("battery_sink")) => {
//                     grid_to_battery += flow
//                 }
//                 (Vertex::Node("battery_source"), Vertex::Node("charger")) => {
//                     battery_to_charger += flow
//                 }
//                 (Vertex::Node("battery_source"), Vertex::Node("grid_sink")) => {
//                     battery_to_grid += flow
//                 }
//                 _ => {}
//             }
//         }
//     }

//     println!("Solar -[{}]-> Charger", solar_to_charger);
//     println!("Solar -[{}]-> Grid", solar_to_grid);
//     println!("Solar -[{}]-> Battery", solar_to_battery);

//     println!("Grid -[{}]-> Charger", grid_to_charger);
//     println!("Grid -[{}]-> Battery", grid_to_battery);

//     println!("Battery -[{}]-> Charger", battery_to_charger);
//     println!("Battery -[{}]-> Grid", battery_to_grid);
// }
