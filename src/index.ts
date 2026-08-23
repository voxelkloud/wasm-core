// The JS side of the kernels. No generated glue: the Rust exports are plain
// functions over linear memory, so this is a few typed-array views onto fixed
// offsets the module hands back.

/** Layout of one child's result: `[containment, key]`. */
export const RESULT_STRIDE = 2;
/** Layout of one child's box: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
export const BOX_STRIDE = 6;

/**
 * Indices into {@link WasmKernels.params}.
 *
 * One block rather than ten arguments, so a call site writes the scalars it
 * changed and nothing crosses the boundary that did not have to.
 */
export const Param = {
  CamX: 0,
  CamY: 1,
  CamZ: 2,
  /** Closed-form radius shared by all eight children. Ignored when `PerChild`. */
  RadiusChild: 3,
  /** Closed-form geometric error shared by all eight. Ignored when `PerChild`. */
  ErrorChild: 4,
  /** @deprecated Renamed to {@link Param.ErrorChild}. */
  SpacingChild: 4,
  NearFloor: 5,
  Slope: 6,
  ViewportHeightPx: 7,
  OrthoProjFactor: 8,
  Orthographic: 9,
  /**
   * Non-zero switches the child loop from the closed-form pair to
   * {@link WasmKernels.child}. Constant for a whole frame, so it is written once
   * alongside the camera scalars and never per node.
   */
  PerChild: 10,
} as const;

export interface WasmKernels {
  /** 24 f64: six normalised planes, `[nx, ny, nz, d]` each. */
  readonly planes: Float64Array;
  /**
   * 48 f64: up to eight child boxes.
   *
   * Doubles as the input for {@link extractPlanes}, which reads a 16-element
   * clip-from-world matrix from the front of it — scratch the child loop is not
   * using at that point in the frame.
   */
  readonly boxes: Float64Array;
  /** 16 f64: `[containment, key]` per child slot. */
  readonly results: Float64Array;
  /** 11 f64, indexed by {@link Param}. */
  readonly params: Float64Array;
  /**
   * 16 f64: `[geometricError; 8]` then `[boundingRadius; 8]`.
   *
   * Read only when `params[Param.PerChild]` is non-zero. Formats whose LOD
   * quantities ARE a closed form of the level — every octree, so Potree v2,
   * COPC and EPT — leave this untouched and write two scalars per admitted node
   * instead of sixteen.
   */
  readonly child: Float64Array;

  /**
   * Cull and score every child of one popped node.
   *
   * ONE crossing per admitted node rather than one per child: at ~30 ns of
   * arithmetic per child against a comparable crossing cost, per-child calls
   * would be a wash. Returns the bitmask of children that survived the frustum.
   */
  selectChildren(mask: number, parentInside: boolean): number;

  /** Fill {@link planes} from a clip-from-world matrix left in {@link boxes}. */
  extractPlanes(depthZeroToOne: boolean, reversedDepth: boolean): void;
}

interface Exports {
  memory: WebAssembly.Memory;
  planes_ptr(): number;
  boxes_ptr(): number;
  results_ptr(): number;
  params_ptr(): number;
  child_ptr(): number;
  select_children(mask: number, parentInside: number): number;
  extract_planes(zeroToOne: number, reversed: number): void;
}

/**
 * Instantiate the kernels from a compiled module.
 *
 * Takes bytes rather than a path so the same function serves Node, a bundler
 * that inlines the `.wasm`, and a `fetch` in the browser.
 */
export async function loadWasmKernels(
  source: BufferSource | WebAssembly.Module,
): Promise<WasmKernels> {
  const instance =
    source instanceof WebAssembly.Module
      ? await WebAssembly.instantiate(source, {})
      : (await WebAssembly.instantiate(source, {})).instance;

  const e = instance.exports as unknown as Exports;
  const buffer = e.memory.buffer;

  // Views onto static Rust arrays: their addresses never move, and the module
  // never grows its memory, so these are valid for its whole life.
  const planes = new Float64Array(buffer, e.planes_ptr(), 24);
  const boxes = new Float64Array(buffer, e.boxes_ptr(), 48);
  const results = new Float64Array(buffer, e.results_ptr(), 16);
  const params = new Float64Array(buffer, e.params_ptr(), 11);
  const child = new Float64Array(buffer, e.child_ptr(), 16);

  return {
    planes,
    boxes,
    results,
    params,
    child,
    selectChildren: (mask, parentInside) =>
      e.select_children(mask, parentInside ? 1 : 0),
    extractPlanes: (zeroToOne, reversed) =>
      e.extract_planes(zeroToOne ? 1 : 0, reversed ? 1 : 0),
  };
}
