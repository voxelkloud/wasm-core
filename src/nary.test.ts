// Task T1: the differential over a node with more than eight children.
//
// The kernel did not change — it still reads eight box slots and eight
// per-child values. What changed is that the scheduler feeds it BLOCKS: a node
// with twelve children is one full block and a tail of four. That is exactly
// the case where a masking or an offset mistake hides, because an octree can
// never produce it, so the guard belongs here rather than in the view's own
// suite: this is the only place the REAL wasm is loaded.

import { existsSync, readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import {
  createLodScratch,
  createLodSelection,
  resolveLodOptions,
  selectVisible,
} from "@voxelkloud/view/lod";
import type {
  LodCameraState,
  LodScratch,
  LodTreeView,
} from "@voxelkloud/view/lod";
import { loadWasmKernels } from "./index.js";
import type { WasmKernels } from "./index.js";

const WASM = new URL("../dist/voxelkloud_wasm_core.wasm", import.meta.url);
const HAS_WASM = existsSync(WASM);
let k: WasmKernels;

beforeAll(async () => {
  if (!HAS_WASM) return;
  const bytes = readFileSync(WASM);
  k = await loadWasmKernels(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
});

/** Deterministic, so a disagreement is reproducible. */
function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

/**
 * A tile tree with an arbitrary fanout per node and boxes that are NOT an
 * octant partition — a tileset's children overlap, leave gaps, and differ in
 * size, all of which an octree forbids and none of which the scheduler may
 * depend on.
 */
function tileTree(fanout: number, depth: number, seed = 7): LodTreeView {
  const rand = rng(seed);
  const nodes: Record<string, unknown>[] = [];
  const errors: number[] = [];
  const radii: number[] = [];

  function make(
    level: number,
    parent: Record<string, unknown> | undefined,
    minX: number,
    maxX: number,
    error: number,
  ): Record<string, unknown> {
    const index = nodes.length;
    const node: Record<string, unknown> = {
      index,
      name: `t${index}`,
      level,
      minX,
      minY: -1 + rand() * 0.2,
      minZ: -1,
      maxX,
      maxY: 1 + rand() * 0.2,
      maxZ: 1,
      numPoints: 1000 + Math.floor(rand() * 5000),
      childMask: 0,
      children: [] as unknown[],
      parent,
    };
    nodes.push(node);
    errors.push(error);
    radii.push(0.5 * (maxX - minX) + 1);

    if (level < depth) {
      const kids: unknown[] = [];
      const span = (maxX - minX) / fanout;
      for (let c = 0; c < fanout; c++) {
        // Children overlap by a tenth of a span: legal in a tileset, impossible
        // in an octree.
        const lo = minX + c * span - span * 0.05;
        const hi = lo + span * 1.1;
        kids.push(make(level + 1, node, lo, hi, error * 0.5));
      }
      node['children'] = kids;
      node['childMask'] = fanout;
    }
    return node;
  }

  make(0, undefined, 0, 1024, 64);

  return {
    nodeCount: nodes.length,
    root: nodes[0] as never,
    node: (i) => nodes[i] as never,
    geometricErrorAt: (l) => 64 / 2 ** l,
    pointSpacingAt: (l) => 64 / 2 ** l,
    boundingRadiusAt: (l) => 513 / 2 ** l,
    nodeGeometricError: Float64Array.from(errors),
    nodeBoundingRadius: Float64Array.from(radii),
    tryExpandSync: () => true,
    requestExpand: () => {},
  };
}

function camera(scratch: LodScratch, distance: number): LodCameraState {
  const cam: LodCameraState = {
    clipFromAbs: new Float64Array(16),
    camX: 512,
    camY: -distance,
    camZ: 0,
    slope: Math.tan(Math.PI / 6),
    viewportHeightPx: 1080,
    orthographic: false,
    orthoProjFactor: 0,
    nearFloor: 1,
    depthRange: "zero-to-one",
    reversedDepth: false,
  };
  // Six planes containing everything, so the two paths are compared on the
  // child maths and the blocking, not on frustum agreement — which
  // `kernels.test.ts` already covers over 4000 random boxes.
  for (let p = 0; p < 6; p++) {
    scratch.planes[p * 4] = 0;
    scratch.planes[p * 4 + 1] = 0;
    scratch.planes[p * 4 + 2] = 0;
    scratch.planes[p * 4 + 3] = 1e9;
  }
  return cam;
}

describe.skipIf(!HAS_WASM)("N-ary selection: wasm against the TypeScript oracle", () => {
  for (const fanout of [3, 4, 8, 9, 12, 17]) {
    it(`agrees byte for byte at fanout ${fanout}`, () => {
      const tree = tileTree(fanout, fanout > 8 ? 2 : 3);
      for (const distance of [50, 400, 3000]) {
        const sTs = createLodScratch();
        const sWs = createLodScratch();
        const ts = createLodSelection();
        const ws = createLodSelection();
        const opts = resolveLodOptions({ pointBudget: 50_000_000 });

        selectVisible(tree, camera(sTs, distance), opts, sTs, ts);
        selectVisible(tree, camera(sWs, distance), opts, sWs, ws, k);

        const where = `fanout ${fanout}, distance ${distance}`;
        expect(ws.count, where).toBe(ts.count);
        expect(ws.points, where).toBe(ts.points);
        expect(ws.limitedBy, where).toBe(ts.limitedBy);
        expect(ws.maxSelectedLevel, where).toBe(ts.maxSelectedLevel);
        expect(ws.achievedScreenError, where).toBeCloseTo(
          ts.achievedScreenError,
          12,
        );
        // POP ORDER, not just membership: it is the streaming priority and the
        // guarantee that a parent is admitted before its child.
        expect(Array.from(ws.indices.subarray(0, ws.count)), where).toEqual(
          Array.from(ts.indices.subarray(0, ts.count)),
        );
      }
    });
  }

  it("selects something worth comparing", () => {
    // A guard on the guard: if the tree were culled to the root, every
    // assertion above would pass while proving nothing.
    const tree = tileTree(12, 2);
    const s = createLodScratch();
    const out = createLodSelection();
    selectVisible(
      tree,
      camera(s, 400),
      resolveLodOptions({ pointBudget: 50_000_000 }),
      s,
      out,
    );
    expect(out.count).toBeGreaterThan(12);
    expect(out.maxSelectedLevel).toBe(2);
  });
});
