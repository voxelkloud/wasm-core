import { existsSync, readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import {
  Containment,
  classifyAabb,
  extractFrustumPlanes,
} from "@voxelkloud/view/lod";
import { Param, loadWasmKernels } from "./index.js";
import type { WasmKernels } from "./index.js";

let k: WasmKernels;

// `pnpm build` in this package compiles the Rust; a clone without a Rust
// toolchain has no wasm to test. Skipping with the reason beats an ENOENT
// stack that reads like a broken test.
const WASM = new URL("../dist/voxelkloud_wasm_core.wasm", import.meta.url);
const HAS_WASM = existsSync(WASM);
if (!HAS_WASM) {
  console.warn(
    "@voxelkloud/wasm-core: dist/voxelkloud_wasm_core.wasm is missing, so the " +
      "kernel tests are skipped. Build it with `pnpm --filter @voxelkloud/" +
      "wasm-core build`, which needs cargo and the wasm32-unknown-unknown " +
      "target (`rustup target add wasm32-unknown-unknown`).",
  );
}

beforeAll(async () => {
  if (!HAS_WASM) return;
  const bytes = readFileSync(WASM);
  k = await loadWasmKernels(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
});

/** Deterministic pseudo-random, so a failure is reproducible. */
function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

/** A plausible clip-from-world matrix at autzen CRS scale. */
function autzenClip(): Float64Array {
  // Built by hand rather than through three, so this package keeps its
  // dependency graph to @voxelkloud/view/lod, which imports nothing itself.
  const fov = (60 * Math.PI) / 180;
  const f = 1 / Math.tan(fov / 2);
  const aspect = 16 / 9;
  const near = 5;
  const far = 20_000;
  const proj = [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ];
  // A view matrix that translates the camera to a point over the cloud.
  const tx = -637_000;
  const ty = -851_000;
  const tz = -900;
  const view = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1];

  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let i = 0; i < 4; i++) sum += proj[i * 4 + r]! * view[c * 4 + i]!;
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

describe.skipIf(!HAS_WASM)("extract_planes agrees with the TypeScript oracle", () => {
  it.each([
    ["zero-to-one", true],
    ["minus-one-to-one", false],
  ] as const)("matches under %s depth", (range, zeroToOne) => {
    const clip = autzenClip();
    const expected = extractFrustumPlanes(
      clip,
      new Float64Array(24),
      range,
      false,
    );

    k.boxes.set(clip, 0);
    k.extractPlanes(zeroToOne, false);

    for (let i = 0; i < 24; i++) {
      // Exact, not approximate: both implementations do the same operations in
      // the same order in f64, so any drift here is a real divergence.
      expect(k.planes[i]).toBeCloseTo(expected[i]!, 12);
    }
  });

  it("matches under reversed depth", () => {
    const clip = autzenClip();
    const expected = extractFrustumPlanes(
      clip,
      new Float64Array(24),
      "zero-to-one",
      true,
    );
    k.boxes.set(clip, 0);
    k.extractPlanes(true, true);
    for (let i = 0; i < 24; i++) {
      expect(k.planes[i]).toBeCloseTo(expected[i]!, 12);
    }
  });
});

describe.skipIf(!HAS_WASM)("select_children agrees with the TypeScript oracle", () => {
  it("classifies 4000 random boxes identically", () => {
    const clip = autzenClip();
    const planes = extractFrustumPlanes(
      clip,
      new Float64Array(24),
      "minus-one-to-one",
      false,
    );
    k.planes.set(planes);

    const rnd = rng(20260822);
    let compared = 0;

    for (let batch = 0; batch < 500; batch++) {
      // Eight boxes scattered around the cloud, some inside, some straddling,
      // some far outside.
      const boxes: number[] = [];
      for (let c = 0; c < 8; c++) {
        const cx = 637_000 + (rnd() - 0.5) * 8000;
        const cy = 851_000 + (rnd() - 0.5) * 8000;
        const cz = 600 + (rnd() - 0.5) * 3000;
        const h = 5 + rnd() * 400;
        boxes.push(cx - h, cy - h, cz - h, cx + h, cy + h, cz + h);
      }
      k.boxes.set(boxes);

      const camX = 637_000;
      const camY = 851_000;
      const camZ = 900;
      const rChild = 250;
      const spChild = 4.55;
      const nearFloor = 5;
      const slope = Math.tan((60 * Math.PI) / 180 / 2);
      const vp = 2160;

      k.params[Param.CamX] = camX;
      k.params[Param.CamY] = camY;
      k.params[Param.CamZ] = camZ;
      k.params[Param.RadiusChild] = rChild;
      k.params[Param.SpacingChild] = spChild;
      k.params[Param.NearFloor] = nearFloor;
      k.params[Param.Slope] = slope;
      k.params[Param.ViewportHeightPx] = vp;
      k.params[Param.OrthoProjFactor] = 0;
      k.params[Param.Orthographic] = 0;
      // Explicit, not inherited: `k` is shared across this file, so a later
      // per-child test must not be able to reach back and change what this one
      // is measuring.
      k.params[Param.PerChild] = 0;

      const mask = 0xff;
      const survived = k.selectChildren(mask, false);

      for (let c = 0; c < 8; c++) {
        const o = c * 6;
        const want = classifyAabb(
          planes,
          boxes[o]!,
          boxes[o + 1]!,
          boxes[o + 2]!,
          boxes[o + 3]!,
          boxes[o + 4]!,
          boxes[o + 5]!,
        );
        expect(k.results[c * 2]).toBe(want);
        expect(((survived >> c) & 1) === 1).toBe(want !== Containment.Outside);

        if (want !== Containment.Outside) {
          const dx = camX - (boxes[o]! + boxes[o + 3]!) * 0.5;
          const dy = camY - (boxes[o + 1]! + boxes[o + 4]!) * 0.5;
          const dz = camZ - (boxes[o + 2]! + boxes[o + 5]!) * 0.5;
          const d = Math.max(Math.hypot(dx, dy, dz) - rChild, nearFloor);
          const key = spChild * ((0.5 * vp) / (slope * d));
          expect(k.results[c * 2 + 1]).toBeCloseTo(key, 9);
        }
        compared++;
      }
    }
    expect(compared).toBe(4000);
  });

  it("propagates Inside without testing planes", () => {
    // A box far outside the frustum still reports Inside when the parent was,
    // because containment propagation is the caller's guarantee, not a
    // re-derivation.
    k.boxes.set([0, 0, 0, 1, 1, 1], 0);
    k.params[Param.CamX] = 1e6;
    k.params[Param.CamY] = 1e6;
    k.params[Param.CamZ] = 1e6;
    k.params[Param.RadiusChild] = 1;
    k.params[Param.SpacingChild] = 1;
    k.params[Param.NearFloor] = 1;
    k.params[Param.Slope] = 0.577;
    k.params[Param.ViewportHeightPx] = 1080;
    k.params[Param.Orthographic] = 0;
    expect(k.selectChildren(0b1, true)).toBe(0b1);
    expect(k.results[0]).toBe(Containment.Inside);
  });

  it("skips slots the mask does not claim", () => {
    k.boxes.fill(0);
    k.results.fill(-1);
    expect(k.selectChildren(0b0000_0101, true)).toBe(0b0000_0101);
    // Untouched slots keep whatever was there; only claimed ones are written.
    expect(k.results[1 * 2]).toBe(-1);
    expect(k.results[3 * 2]).toBe(-1);
  });

  it("clamps distance at the near floor instead of exploding", () => {
    // A camera inside the box: the un-clamped distance is negative, and a
    // sentinel would make every such node tie.
    k.planes.fill(0);
    k.boxes.set([-10, -10, -10, 10, 10, 10], 0);
    k.params[Param.CamX] = 0;
    k.params[Param.CamY] = 0;
    k.params[Param.CamZ] = 0;
    k.params[Param.RadiusChild] = 100;
    k.params[Param.SpacingChild] = 2;
    k.params[Param.NearFloor] = 5;
    k.params[Param.Slope] = 0.577;
    k.params[Param.ViewportHeightPx] = 1080;
    k.params[Param.Orthographic] = 0;
    k.selectChildren(0b1, true);
    const expected = 2 * ((0.5 * 1080) / (0.577 * 5));
    expect(k.results[1]).toBeCloseTo(expected, 9);
    expect(Number.isFinite(k.results[1]!)).toBe(true);
  });
});

describe.skipIf(!HAS_WASM)("select_children: the per-child block (A3)", () => {
  const camX = 637_000;
  const camY = 851_000;
  const camZ = 900;
  const nearFloor = 5;
  const slope = Math.tan((60 * Math.PI) / 180 / 2);
  const vp = 2160;

  /** Eight boxes near the cloud centre, all comfortably inside the frustum. */
  function eightBoxes(rnd: () => number): number[] {
    const boxes: number[] = [];
    for (let c = 0; c < 8; c++) {
      const cx = 637_000 + (rnd() - 0.5) * 1500;
      const cy = 851_000 + (rnd() - 0.5) * 1500;
      const cz = 600 + (rnd() - 0.5) * 400;
      const h = 20 + rnd() * 120;
      boxes.push(cx - h, cy - h, cz - h, cx + h, cy + h, cz + h);
    }
    return boxes;
  }

  function writeCamera(): void {
    k.params[Param.CamX] = camX;
    k.params[Param.CamY] = camY;
    k.params[Param.CamZ] = camZ;
    k.params[Param.NearFloor] = nearFloor;
    k.params[Param.Slope] = slope;
    k.params[Param.ViewportHeightPx] = vp;
    k.params[Param.OrthoProjFactor] = 0;
    k.params[Param.Orthographic] = 0;
  }

  it("scores each child from ITS OWN error and radius", () => {
    const clip = autzenClip();
    const planes = extractFrustumPlanes(
      clip,
      new Float64Array(24),
      "minus-one-to-one",
      false,
    );
    k.planes.set(planes);
    const rnd = rng(20260823);
    let compared = 0;

    for (let batch = 0; batch < 200; batch++) {
      const boxes = eightBoxes(rnd);
      k.boxes.set(boxes);
      writeCamera();
      // Deliberately absurd closed-form values: if the kernel reads them while
      // PerChild is set, every key below is wrong by orders of magnitude.
      k.params[Param.RadiusChild] = 1e9;
      k.params[Param.ErrorChild] = 1e9;
      k.params[Param.PerChild] = 1;

      // Eight DISTINCT pairs, so a kernel that broadcast slot 0 to all eight —
      // the obvious way to get this wrong — fails.
      const err: number[] = [];
      const rad: number[] = [];
      for (let c = 0; c < 8; c++) {
        err.push(0.5 + c * 1.25);
        rad.push(80 + c * 37);
        k.child[c] = err[c]!;
        k.child[8 + c] = rad[c]!;
      }

      const survived = k.selectChildren(0xff, false);

      for (let c = 0; c < 8; c++) {
        const o = c * 6;
        const want = classifyAabb(
          planes,
          boxes[o]!, boxes[o + 1]!, boxes[o + 2]!,
          boxes[o + 3]!, boxes[o + 4]!, boxes[o + 5]!,
        );
        expect(k.results[c * 2]).toBe(want);
        expect(((survived >> c) & 1) === 1).toBe(want !== Containment.Outside);
        if (want !== Containment.Outside) {
          const dx = camX - (boxes[o]! + boxes[o + 3]!) * 0.5;
          const dy = camY - (boxes[o + 1]! + boxes[o + 4]!) * 0.5;
          const dz = camZ - (boxes[o + 2]! + boxes[o + 5]!) * 0.5;
          const d = Math.max(Math.hypot(dx, dy, dz) - rad[c]!, nearFloor);
          expect(k.results[c * 2 + 1]).toBeCloseTo(
            err[c]! * ((0.5 * vp) / (slope * d)),
            9,
          );
        }
        compared++;
      }
    }
    k.params[Param.PerChild] = 0;
    expect(compared).toBe(1600);
  });

  it("ignores the per-child block entirely when the flag is clear", () => {
    // The cost argument for the whole design: an octree writes two scalars and
    // never touches `child`. Stale bytes left there by an earlier frame — or by
    // the test above — must not reach the result.
    const clip = autzenClip();
    const planes = extractFrustumPlanes(
      clip,
      new Float64Array(24),
      "minus-one-to-one",
      false,
    );
    k.planes.set(planes);
    const boxes = eightBoxes(rng(7));
    k.boxes.set(boxes);
    writeCamera();

    const rChild = 250;
    const eChild = 4.55;
    k.params[Param.RadiusChild] = rChild;
    k.params[Param.ErrorChild] = eChild;
    k.params[Param.PerChild] = 0;
    for (let c = 0; c < 16; c++) k.child[c] = -12345;

    const survived = k.selectChildren(0xff, false);

    for (let c = 0; c < 8; c++) {
      if (((survived >> c) & 1) === 0) continue;
      const o = c * 6;
      const dx = camX - (boxes[o]! + boxes[o + 3]!) * 0.5;
      const dy = camY - (boxes[o + 1]! + boxes[o + 4]!) * 0.5;
      const dz = camZ - (boxes[o + 2]! + boxes[o + 5]!) * 0.5;
      const d = Math.max(Math.hypot(dx, dy, dz) - rChild, nearFloor);
      expect(k.results[c * 2 + 1]).toBeCloseTo(eChild * ((0.5 * vp) / (slope * d)), 9);
    }
  });
});
