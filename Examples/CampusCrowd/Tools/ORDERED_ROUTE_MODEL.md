# CampusCrowd ordered-route encoder and decoder

The model takes a source landmark and a destination stop and returns an
ordered list of landmark IDs. A graph neural network encodes the campus
topology, and a gated recurrent unit (GRU) selects one outgoing edge per hop.
The Python route command uses Dijkstra from the original source if the
decoder reaches a dead end or exceeds its hop limit. The Unreal demo runs
the separate [ETA-guided A* planner](NAVIGATION_RL.md).

[Implementation](train_ordered_routes.py) ·
[Graph asset](../Content/Campus/navigation_landmarks.json) ·
[Stop metadata](../Content/Campus/navigation_stops.json)

## Graph and inputs

The trained graph has **2,556 landmarks**, **14,683 undirected walkable
links**, and **197 destination stops**. The stops comprise four campus
gateways and 193 mapped building entrances. The maximum landmark degree
is 45, with a mean of 11.49. Graph edges carry lengths in metres. The loader
recovers lengths from the exported edge times using the graph's reference
walking pace of 1.4 m/s.

Each landmark starts with five features in this order.

| Features | Encoding |
| --- | --- |
| Local x and y coordinates | Metres divided by 100 |
| Landmark type | One-hot flags for ordinary waypoint, campus gateway, and building entrance |

For example, a building entrance at local coordinates (200, -50) metres
starts as `[2.0, -0.5, 0, 0, 1]`. A source can be any landmark. The
destination argument is a stop index, which the loader resolves to a
landmark ID through the graph's stop table.

## Graph encoder

The encoder starts with a learned `Linear(5, 128)` layer. Each of the next
three layers aggregates adjacent landmarks and updates the current embedding.
For an edge of length `L[i,j]` metres, the adjacency weight is proportional
to `1 / (1 + L[i,j] / 30)`. The weights are normalized separately for each
landmark so its outgoing weights sum to one. This gives nearby neighbors
more influence.

Let `H` be the current `[N, 128]` embedding matrix and `A` the
row-normalized sparse adjacency. A layer computes

```text
M = A @ H
H_next = LayerNorm(H + SiLU(Linear_256_to_128(concat(H, M))))
```

The concatenation `concat(H, M)` combines each landmark's own vector and its
weighted neighbor average. Each layer has its own linear and normalization
parameters. After three layers, a separate `Linear(128, 64)` projection
produces a candidate key for each landmark. The encoder returns the
`[N, 128]` embeddings and the `[N, 64]` keys. They depend on the
graph and trained weights, so an inference session can reuse them across
sources and destinations. The single-route CLI computes them once per
invocation.

## Autoregressive route decoder

1. Concatenate the source and goal embeddings, apply `Linear(256, 128)`
   and `tanh`, and use the result as the initial hidden state.
2. At each hop, feed the concatenated current-landmark and goal embeddings
   into a `GRUCell(256, 128)` with the previous hidden state.
3. Concatenate the new hidden state, current embedding, and goal embedding.
   A `Linear(384, 64)` layer produces the query. For each outgoing
   neighbor, take the dot product of the query and its cached 64-dimensional
   key, scaled by one eighth because the keys have dimension 64.
4. Add a geometric score from an `8 → 64 → 1` multilayer perceptron with a
   SiLU hidden activation. Its eight inputs are the two components of the
   current-to-goal vector, the two components of the current-to-neighbor
   vector, the current and neighbor straight-line distances to the goal,
   the reduction in straight-line distance, and the walkable edge length.
   Geometry uses 100-metre units.
5. Mask padded entries and visited landmarks. Greedily choose the
   highest-scoring eligible neighbor, append it, and continue. Arrival
   terminates the route. With the selected checkpoint, decoding stops after
   at most 128 hops or as soon as no eligible edge remains.

The returned sequence starts at the requested source. Every transition is
a graph edge, and the visited mask prevents cycles. The greedy decoder can
still take a detour or end in a dead end. On failure, the CLI discards the
partial route and runs Dijkstra from the original source. Evaluation
reconstructs the equivalent fallback from a precomputed Dijkstra next-hop
table. The model has **314,881 trainable parameters** and uses shared
scoring weights across neighbors, so the output layer has no fixed
landmark vocabulary.

## Tensor layout and batching

| Tensor | Shape on this graph | Purpose |
| --- | --- | --- |
| Node features | `[2,556, 5]` | Input to the encoder |
| Sparse adjacency | `[2,556, 2,556]` with 29,366 nonzero entries | Neighbor averaging |
| Landmark embeddings and keys | `[2,556, 128]` and `[2,556, 64]` | Cached graph representation |
| Padded neighbor IDs, lengths, and validity masks | `[2,556, 45]` each | Gather outgoing edges for the current landmarks |
| Per-route GRU state and visited mask | `[B, 128]` and `[B, 2,556]` | Track decoder state and exclude revisits |
| Candidate features and scores per hop | `[B, 45, 8]` and `[B, 45]` | Score eligible edges and select the next hop |
| Predicted path buffer at inference | `[B, 129]` | Store the source and up to 128 subsequent landmarks |

The encoder uses sparse matrix multiplication across all landmarks at each
layer. The decoder vectorizes the GRU across routes and scores all padded
candidate slots together using tensor operations at each hop. The hop loop is sequential.
Finished or blocked routes stop writing to the path buffer. Their rows still
consume decoder work until the batch stops. Training recomputes graph
embeddings once per optimizer step so gradients can update the encoder.

## Supervision and checkpoint selection

Reverse Dijkstra from each stop computes exact remaining graph distance
`D(v, g)` from each landmark `v` to goal `g`. At each training hop, the
teacher assigns each eligible neighbor `j` the cost `L[i,j] + D(j, g)`.
Hops within 0.001 m of the best eligible cost are all positive labels.
The loss is the negative log of the total softmax probability assigned to
those hops. After a model-selected move, the teacher applies the same rule
with the updated visited mask.

Training samples landmark-to-stop requests, then unrolls short windows.
Some training steps follow the model's greedy move to expose the decoder to
its own errors. Refinement also penalizes expected extra path cost at the
next hop. For each eligible neighbor, it subtracts the best candidate
cost, caps the difference at 50 m, divides by 10 m, and weights the result
by the model's probability of choosing that neighbor. The resulting
penalty has a weight of 0.45 in the refinement loss.

| Setting | Initial training | Refinement |
| --- | ---: | ---: |
| Epochs run | 24 | 36 |
| Batches per epoch | 48 | 48 |
| Requests per batch | 128 | 128 |
| Training hop window | 5 | 8 |
| Initial AdamW learning rate | 0.002 | 0.0009 |
| Probability of following a model move | Ramps from 0% to 20% | 20% |
| Extra-cost penalty weight | 0 | 0.45 |

Both stages used seed 41, cosine learning-rate decay to 15% of the initial
rate, and gradient clipping at norm 1.0. The split uses seed 317 and keeps
all entrances for a building together. There are 149 training, 16
validation, and 32 test destination stops. The four gateways are training
destinations. All phases have access to the same campus topology. These
results measure routing to held-out destinations on that topology.
Generalization to a changed graph is unmeasured.

Checkpoint selection scores greedy rollouts on 256 validation
stop-to-stop requests. The score is the fraction completing without
fallback divided by the 95th percentile successful-route cost ratio.
The number of routes within 1% of the shortest distance breaks ties.
The selected checkpoint is from refinement epoch 19. It completed
240 of 256 validation requests without fallback, with a p95 cost ratio
of 1.392.

## Held-out route quality

Both test sets use destinations at held-out buildings. Landmark-to-stop
requests sample from all distinct source landmarks and destination stops.
Stop-to-stop requests sample distinct stops at different buildings or from
a campus gateway to a held-out building. Each test set has 1,200 requests.

| Metric | Landmark to stop | Stop to stop |
| --- | ---: | ---: |
| Completed by the decoder | 1,106 / 1,200 (92.2%) | 1,094 / 1,200 (91.2%) |
| Dijkstra fallbacks | 94 (7.8%) | 106 (8.8%) |
| Decoder routes within 1% of shortest graph distance | 253 | 268 |
| Median successful decoder route / shortest route | 1.035× | 1.033× |
| p95 successful decoder route / shortest route | 1.325× | 1.351× |
| Maximum successful decoder route / shortest route | 3.522× | 8.406× |

Distances sum the exported edge lengths. The ratio statistics include only
routes the decoder completed on its own. Fallback completed the remaining
requests. Route quality here is graph distance, with walking simulation
and shortcut refinement outside the measured scope.

## CPU inference performance

Measurements used an AMD Ryzen 9 9950X with PyTorch 2.15.0+cpu and four
PyTorch threads.

| Batched measurement | Landmark to stop | Stop to stop |
| --- | ---: | ---: |
| Encode the full graph once | 2.50 ms | 2.84 ms |
| Decode all 1,200 requests in batches of up to 128 | 280.21 ms | 291.53 ms |
| Amortized decode time per request | 0.234 ms | 0.243 ms |

Batched decoding times exclude graph encoding, checkpoint loading, and
fallback reconstruction. These are throughput figures. Single-request
latency was measured separately on the first 256 saved stop-to-stop test
requests, in file order, after 16 warm-up requests for each planner.

| Python planner, one request at a time | Median | p95 |
| --- | ---: | ---: |
| Encoder/decoder with cached graph embeddings and Dijkstra fallback | 1.963 ms | 4.021 ms |
| Straight-line A* with cached goal heuristics | 0.281 ms | 1.338 ms |
| Dijkstra search | 1.325 ms | 2.287 ms |

The single-request test timed in-process route planning and landmark
sequence reconstruction. It included all 26 fallback searches in the
neural planner's sample, with setup and graph encoding excluded. On this
workload, straight-line A* had the lowest single-request latency. Unreal
runtime latency and graphics frame time were outside the measured scope.

## Reproduce and inspect

Run from `Examples/CampusCrowd` in the devcontainer.

```bash
base=Saved/NavigationRL/ordered
python Tools/train_ordered_routes.py --output "$base"
python Tools/train_ordered_routes.py --warm-start "$base/ordered_routes.pt" \
  --output "$base/refinement" --epochs 36 --window 8 --max-steps 128 \
  --learning-rate .0009 --regret-weight .45
python Tools/train_ordered_routes.py --output "$base/refinement" --route 2541 7
```

The example goes from landmark 2541 to stop index 7, which resolves to
landmark 2366. The selected model returns a 16-landmark sequence with no
fallback. Checkpoints store a hash of the landmark graph, and the CLI
checks it before loading the model.

The training commands generate the following local artifacts under
`Saved/NavigationRL/ordered/refinement`.

- [Trained PyTorch checkpoint](../Saved/NavigationRL/ordered/refinement/ordered_routes.pt)
- [Training history and evaluation report](../Saved/NavigationRL/ordered/refinement/report.json)
- [Landmark-to-stop test routes](../Saved/NavigationRL/ordered/refinement/held_out_landmark_to_stop_routes.json)
- [Stop-to-stop test routes](../Saved/NavigationRL/ordered/refinement/held_out_stop_to_stop_routes.json)
