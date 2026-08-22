//! Numeric kernels for the voxelkloud LOD scheduler.
//!
//! No `wasm-bindgen`. These are pure functions over `f64` in linear memory, so
//! the JS side reads and writes fixed offsets through a `Float64Array` view and
//! the module needs no generated glue at all — smaller, and one fewer build
//! tool. `cargo build --target wasm32-unknown-unknown --release` is the whole
//! build.
//!
//! GRANULARITY IS THE POINT. Porting `classify_aabb` alone would cost one
//! JS->wasm crossing per CHILD, and at ~30 ns of arithmetic against a crossing
//! of comparable cost that is a wash or worse. `select_children` instead does
//! every child of one popped node in a single call: the frustum test, the
//! distance, the projection factor and the priority key for up to eight boxes.
//! That is one crossing per ADMITTED NODE, which is the unit the measured cost
//! scales with (1.2-2.6 us/node in TypeScript).
//!
//! Everything is f64. On autzen a plane constant is of order 6.4e5 and
//! `n·p + d` is a difference of two such values: ~1e-10 in f64 and ~0.05 m in
//! f32, which is a whole point spacing at level 6.

#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // Nothing here can panic: every index is a compile-time constant and there
    // is no allocation. This exists only to satisfy `no_std`.
    loop {}
}

/// Six frustum planes: `[nx, ny, nz, d] * 6`, normalised, world space.
static mut PLANES: [f64; 24] = [0.0; 24];

/// Up to eight child boxes: `[minX, minY, minZ, maxX, maxY, maxZ] * 8`.
static mut BOXES: [f64; 48] = [0.0; 48];

/// Per-child output: `[containment, key] * 8`.
static mut RESULTS: [f64; 16] = [0.0; 16];

/// Scalars the child loop needs, in one block so JS writes them in one go:
/// `[camX, camY, camZ, rChild, spChild, nearFloor, slope, halfViewportPx,
///   orthoProjFactor, orthographic]`.
static mut PARAMS: [f64; 10] = [0.0; 10];

const OUTSIDE: f64 = 0.0;
const INTERSECTING: f64 = 1.0;
const INSIDE: f64 = 2.0;

#[no_mangle]
pub extern "C" fn planes_ptr() -> *const f64 {
    &raw const PLANES as *const f64
}

#[no_mangle]
pub extern "C" fn boxes_ptr() -> *const f64 {
    &raw const BOXES as *const f64
}

#[no_mangle]
pub extern "C" fn results_ptr() -> *const f64 {
    &raw const RESULTS as *const f64
}

#[no_mangle]
pub extern "C" fn params_ptr() -> *const f64 {
    &raw const PARAMS as *const f64
}

/// Classify one box against the six planes.
///
/// p-vertex / n-vertex: the p-vertex is the corner furthest along the normal,
/// the n-vertex the one furthest against it. A p-vertex behind any plane means
/// the box is entirely outside; any n-vertex behind means it straddles.
#[inline(always)]
fn classify(planes: &[f64; 24], b: &[f64], o: usize) -> f64 {
    let min_x = b[o];
    let min_y = b[o + 1];
    let min_z = b[o + 2];
    let max_x = b[o + 3];
    let max_y = b[o + 4];
    let max_z = b[o + 5];

    let mut intersecting = false;
    let mut p = 0usize;
    while p < 24 {
        let nx = planes[p];
        let ny = planes[p + 1];
        let nz = planes[p + 2];
        let d = planes[p + 3];

        let px = if nx > 0.0 { max_x } else { min_x };
        let py = if ny > 0.0 { max_y } else { min_y };
        let pz = if nz > 0.0 { max_z } else { min_z };
        if nx * px + ny * py + nz * pz + d < 0.0 {
            return OUTSIDE;
        }

        let qx = if nx > 0.0 { min_x } else { max_x };
        let qy = if ny > 0.0 { min_y } else { max_y };
        let qz = if nz > 0.0 { min_z } else { max_z };
        if nx * qx + ny * qy + nz * qz + d < 0.0 {
            intersecting = true;
        }
        p += 4;
    }

    if intersecting {
        INTERSECTING
    } else {
        INSIDE
    }
}

/// Cull and score every child of one popped node.
///
/// `mask` is the octant occupancy bitfield — bit `c` set means slot `c` of
/// `BOXES` holds a real child. `parent_inside` skips the plane test entirely,
/// because a box fully inside the frustum has every descendant inside too.
///
/// Writes `[containment, key]` per slot into `RESULTS` and returns the bitmask
/// of children that were NOT culled. The caller still owns the heap and the
/// target-spacing cut: this kernel decides visibility and priority, nothing
/// about traversal order.
#[no_mangle]
pub extern "C" fn select_children(mask: u32, parent_inside: u32) -> u32 {
    let planes = unsafe { &*(&raw const PLANES) };
    let boxes = unsafe { &*(&raw const BOXES) };
    let params = unsafe { &*(&raw const PARAMS) };
    let results = unsafe { &mut *(&raw mut RESULTS) };

    let cam_x = params[0];
    let cam_y = params[1];
    let cam_z = params[2];
    let r_child = params[3];
    let sp_child = params[4];
    let near_floor = params[5];
    let slope = params[6];
    let half_vp = params[7];
    let ortho_pf = params[8];
    let orthographic = params[9] != 0.0;

    let mut out_mask = 0u32;

    let mut c = 0usize;
    while c < 8 {
        if (mask >> c) & 1 == 0 {
            c += 1;
            continue;
        }
        let o = c * 6;

        let containment = if parent_inside != 0 {
            INSIDE
        } else {
            classify(planes, boxes, o)
        };
        if containment == OUTSIDE {
            results[c * 2] = OUTSIDE;
            results[c * 2 + 1] = 0.0;
            c += 1;
            continue;
        }

        let dx = cam_x - (boxes[o] + boxes[o + 3]) * 0.5;
        let dy = cam_y - (boxes[o + 1] + boxes[o + 4]) * 0.5;
        let dz = cam_z - (boxes[o + 2] + boxes[o + 5]) * 0.5;
        // A finite clamp, not a sentinel: it keeps the key monotone in level, so
        // a parent always outranks its own children even when the camera is
        // inside the box.
        let mut dist = sqrt(dx * dx + dy * dy + dz * dz) - r_child;
        if dist < near_floor {
            dist = near_floor;
        }

        let pf = if orthographic {
            ortho_pf
        } else {
            (0.5 * half_vp) / (slope * dist)
        };

        results[c * 2] = containment;
        results[c * 2 + 1] = sp_child * pf;
        out_mask |= 1 << c;
        c += 1;
    }

    out_mask
}

/// `no_std` has no `f64::sqrt`, and wasm has a native instruction for it.
#[inline(always)]
fn sqrt(x: f64) -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        core::arch::wasm32::f64x2_extract_lane::<0>(core::arch::wasm32::f64x2_sqrt(
            core::arch::wasm32::f64x2_splat(x),
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Host builds exist only so `cargo test` can run; precision here is not
        // load-bearing.
        let mut g = x;
        let mut i = 0;
        while i < 40 && g > 0.0 {
            g = 0.5 * (g + x / g);
            i += 1;
        }
        g
    }
}

/// Extract six normalised planes from a column-major clip-from-world matrix.
///
/// The matrix is read from `BOXES[0..16]`, which is scratch the caller is not
/// using at this point, and the planes land in `PLANES`. Order matches three's
/// `Frustum`: right, left, bottom, top, far, near.
///
/// `depth_zero_to_one` selects the WebGPU convention. There is no default: the
/// wrong one does not fail loudly, it misplaces the near plane and the bug
/// presents as flicker close to the camera.
#[no_mangle]
pub extern "C" fn extract_planes(depth_zero_to_one: u32, reversed: u32) {
    let m = unsafe { &*(&raw const BOXES) };
    let planes = unsafe { &mut *(&raw mut PLANES) };

    let (me0, me1, me2, me3) = (m[0], m[1], m[2], m[3]);
    let (me4, me5, me6, me7) = (m[4], m[5], m[6], m[7]);
    let (me8, me9, me10, me11) = (m[8], m[9], m[10], m[11]);
    let (me12, me13, me14, me15) = (m[12], m[13], m[14], m[15]);

    set_plane(planes, 0, me3 - me0, me7 - me4, me11 - me8, me15 - me12);
    set_plane(planes, 1, me3 + me0, me7 + me4, me11 + me8, me15 + me12);
    set_plane(planes, 2, me3 + me1, me7 + me5, me11 + me9, me15 + me13);
    set_plane(planes, 3, me3 - me1, me7 - me5, me11 - me9, me15 - me13);

    if reversed != 0 {
        set_plane(planes, 4, me2, me6, me10, me14);
        set_plane(planes, 5, me3 - me2, me7 - me6, me11 - me10, me15 - me14);
    } else if depth_zero_to_one != 0 {
        set_plane(planes, 4, me3 - me2, me7 - me6, me11 - me10, me15 - me14);
        set_plane(planes, 5, me2, me6, me10, me14);
    } else {
        set_plane(planes, 4, me3 - me2, me7 - me6, me11 - me10, me15 - me14);
        set_plane(planes, 5, me3 + me2, me7 + me6, me11 + me10, me15 + me14);
    }
}

#[inline(always)]
fn set_plane(planes: &mut [f64; 24], i: usize, nx: f64, ny: f64, nz: f64, d: f64) {
    let len = sqrt(nx * nx + ny * ny + nz * nz);
    let k = if len == 0.0 { 0.0 } else { 1.0 / len };
    let o = i * 4;
    planes[o] = nx * k;
    planes[o + 1] = ny * k;
    planes[o + 2] = nz * k;
    planes[o + 3] = d * k;
}
