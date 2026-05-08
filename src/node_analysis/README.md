# Hex Maze Analysis — Column Reference

`hex_maze_analysis.py` reads raw trial Excel files and appends computed columns to each row.
This document explains exactly how every output value is calculated.

---

## Maze & Graph Structure

The maze has **96 nodes** across 4 islands (nodes 101–124, 201–224, 301–324, 401–424) plus
2 homeboxes (501, 502). Islands are connected by bridges; the script builds two graphs at startup:

- **Node graph** (`G`) — full 96-node maze, used for all node-level distances.
- **Island graph** (`_ISLAND_G`) — 4-node graph where each node is an island (1–4), edges are bridge connections.

All-pairs shortest paths are pre-computed once at startup for both graphs.

---

## Required Input Columns

| Column | Description |
|---|---|
| `path_to_reach` | Comma-separated node IDs representing the rat's full path (e.g. `101,102,103`) |
| `start_node_n` | Node ID where the trial began |
| `goal_node_n` | Node ID of the food location |
| `start_island_n` | Island number (1–4) containing the start node |
| `goal_island_n` | Island number (1–4) containing the goal node |
| `seq_islands` | Comma-separated sequence of islands visited (can repeat, e.g. `1,2,2,3`) |
| `exclude_trial` | `0` = include in Step 2; any other value = skip Step 2 for this row |
| `comment` | Optional note; used as the `flag` message when `path_to_reach` is empty |

---

## Computed Columns

### Distance Metrics

**`distance_start_goal_island`**
> Shortest island-to-island distance + 1.

```
distance_start_goal_island = island_graph_distance(start_island, goal_island) + 1
```

The `+ 1` converts from edge-count to step-count (a rat starting and ending on the same
island has a distance of 1, not 0).

---

**`distance_start_goal_nodes`**
> Shortest node-to-node distance + 1.

```
distance_start_goal_nodes = node_graph_distance(start_node, goal_node) + 1
```

Same convention as above — counts the number of nodes on the optimal path, including start.

---

### Path Length Metrics

These three columns describe how far the rat actually traveled in different units.

**`path_length_start_goal_nodes_node_hit`**
> Total nodes visited: `len(path_to_reach)`.

Every node the rat stepped on, including start and goal, counted once per visit.

---

**`path_length_start_goal_island_node_hit`**
> Total island entries: `len(seq_islands)`.

Counts every island entry in `seq_islands`, including repeat visits to the same island.

---

**`path_length_start_goal_island_island_hit`**
> Unique islands visited: `len(set(seq_islands))`.

Counts each island at most once regardless of how many times it was entered.

---

### Normalized Path Lengths

Each normalized metric divides a raw path length by the corresponding optimal distance,
giving a score where **1.0 = optimal, > 1.0 = detour**.

**`norm_path_length_start_goal_nodes_node_hit`**
```
path_length_start_goal_nodes_node_hit / distance_start_goal_nodes
```

**`norm_path_length_start_goal_island_node_hit`**
```
path_length_start_goal_island_node_hit / distance_start_goal_island
```

**`norm_path_length_start_goal_island_island_hit`**
```
path_length_start_goal_island_island_hit / distance_start_goal_island
```

All three are `NaN` if the denominator is 0.

---

### Step 1 — Core Behavioral Metrics

**`shortest_path`**
> Minimum number of edges (hops) between `start_node` and `goal_node` in the maze graph.

This is the graph-theoretic shortest path, independent of what the rat actually did.
It is the denominator for several performance metrics below.

---

**`n_nodes_visited`**
> `len(path_to_reach)` — total nodes in the rat's path, including start and any revisits.

---

**`food_reached`**
> `1` if `goal_node` appears among the last two nodes of the path; `0` otherwise.

The two-node window accommodates minor position jitter where the rat steps onto goal then
immediately back one node before the trial ends.

---

**`eat_on_1_encounter`**
> `1` if the **last** node of the path equals `goal_node`; `0` otherwise.

Stricter than `food_reached` — requires the rat to finish exactly on the goal node with
no backtrack recorded at the end.

---

**`dist_tra`**
> Actual distance traveled in edges.

```
dist_tra = n_nodes_visited - 1   if food_reached == 1
dist_tra = 99                    if food_reached == 0
```

The sentinel value `99` marks incomplete trials so they stand out in analysis without
producing misleading ratios.

---

**`dt_rel_sp`** *(relative distance)*
> How many times longer than optimal the rat's path was.

```
dt_rel_sp = dist_tra / shortest_path
```

- `1.0` — rat took the shortest possible route.
- `> 1.0` — rat took detours; e.g. `2.0` means twice as long.
- `NaN` if `shortest_path == 0` (start == goal).

---

**`dt_min_sp`** *(extra steps)*
> Number of steps beyond the optimal path.

```
dt_min_sp = dist_tra - shortest_path
```

- `0` — no detour.
- Positive integers indicate how many extra hops were taken.

---

**`dir_run_mat_perf`** *(direct run)*
> `1` if the rat reached the goal AND took the optimal path; `0` otherwise.

```
dir_run_mat_perf = 1   if food_reached == 1 AND dist_tra == shortest_path
dir_run_mat_perf = 0   otherwise
```

---

**`node_choices_binary`**
> Comma-separated string of `0`/`1` values, one per step in the path.

At each step from node `curr` to node `next`, the script checks whether `next` minimized
the remaining distance to the goal among all neighbors of `curr`:

```
min_sp = min(node_graph_distance(neighbor, goal_node) for neighbor in neighbors(curr))
choice = 1  if node_graph_distance(next, goal_node) == min_sp  else  0
```

A `0` is also appended if `next` is not a valid neighbor of `curr` (path continuity error).

Example: `1,1,0,1` means 3 of 4 steps were optimal choices.

---

**`perc_correct_choices`**
> Percentage of steps in `node_choices_binary` that were `1`.

```
perc_correct_choices = (sum(node_choices_binary) / (n_nodes_visited - 1)) * 100
```

`NaN` if the path has only one node. Range is 0–100.

---

### Step 2 — Goal-Island Entry Analysis

Step 2 is **skipped** for rows where `exclude_trial != 0`.

These metrics measure the rat's performance starting from the moment it last crossed a
bridge into the goal island.

A "bridge crossing" is detected as any step where the absolute difference between
consecutive node IDs is ≥ 50 (node numbering jumps by ~100 between islands, so
cross-island moves produce large differences; intra-island moves are small).
The **last** such crossing in the path is used as the island-entry point.

---

**`isl_node_in`**
> The node the rat was at when it last crossed into the goal island.

This is `path[index_of_last_large_jump]` — the node *before* the crossing step.

---

**`isl_short_path`**
> Optimal distance from the island-entry node to the goal, + 1.

```
isl_short_path = node_graph_distance(isl_node_in, goal_node) + 1
```

---

**`isl_dt_trav`**
> Number of nodes from the island-entry point to the end of the path.

```
isl_dt_trav = len(path[index_of_last_large_jump:])
```

---

**`perf_in_island`**
> Performance ratio within the goal island: how many times longer than optimal.

```
perf_in_island = isl_dt_trav / isl_short_path
```

- `1.0` — optimal from entry.
- `> 1.0` — detours taken inside the island.
- `NaN` if `isl_short_path == 0`.

---

### Placeholder Columns (headers only, not computed here)

These columns are added to the output sheet with empty values as reserved slots for
downstream manual entry or other analysis scripts:

`Diff_Lat_reach_eat`, `goal_island_i_e`, `start_island_i_e`, `dir_run_mat_lat`,
`drug`, `number_times_drug_infused`, `lg-DT_REL_SP`, `lg10-DT_REL_SP`, `lg_perf_I`,
`Project`, `Training_order`, `Implant`, `Number_of_goal_locations`

---

### Error Tracking

**`flag`**
> Empty string for successfully processed rows. Non-empty values indicate a problem.

Rows are flagged (and highlighted red in the output Excel) when:
- `path_to_reach` is empty or missing → the `comment` column value is used as the flag
  message (or `"unknown error"` if comment is also empty).
- Any exception is raised during computation (e.g. unknown node ID, start-node mismatch,
  parse error) → the exception message is stored here.

Flagged rows still appear in the output; all computed columns for that row are left blank.

---

## Running the Script

```bash
python hex_maze_analysis.py \
    --input_folder  /path/to/ip1 \
    --output_folder /path/to/op1
```

All `.xlsx` files in `--input_folder` are processed. Each produces a `*_results.xlsx`
file in `--output_folder` — a copy of the original with the computed columns written in
place (or appended if the columns do not yet exist).
