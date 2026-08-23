# @voxelkloud/wasm-core

Rust LOD kernels for [voxelkloud](../../README.md), as raw wasm over linear
memory. 3,603 bytes, 1,467 gzipped.

```sh
npm install @voxelkloud/wasm-core
```

```ts
import { loadWasmKernels } from "@voxelkloud/wasm-core";
import { selectVisible } from "@voxelkloud/view/lod";

const kernels = await loadWasmKernels(wasmBytes);
selectVisible(tree, cam, options, scratch, out, kernels);
```

No wasm-bindgen. These are pure f64 functions, so JS reads and writes fixed
offsets through `Float64Array` views and the module needs no generated glue.

OPT-IN, and not wired in by default. `@voxelkloud/view/lod` declares the kernel
surface structurally rather than importing it, so the scheduler keeps its
zero-dependency import graph and the TypeScript path stays the
differential-test oracle.

MEASURED, warm, against that oracle: 1.1x to 1.45x on selection, which is about
0.6 ms of a 16.6 ms frame at the most extreme setting. At the shipped defaults
selection costs 0.1-0.5 ms, where this is irrelevant. Reach for it only after
measuring that selection is your bottleneck.

Building from source needs cargo and `rustup target add
wasm32-unknown-unknown`.

MIT.
