import { readFileSync } from "node:fs";
import { createHierarchy, parsePointCloudSource } from "@voxelkloud/loader";
import {
  createLodScratch, createLodSelection, extractFrustumPlanes,
  resolveLodOptions, selectVisible,
} from "@voxelkloud/view/lod";
import { existsSync } from "node:fs";
import { it } from "vitest";
import { loadWasmKernels } from "./index.js";

// Resolved from this file, not hardcoded: an absolute path pinned to one
// machine's checkout makes the guard silently skip everywhere else, which reads
// as "no data" rather than as "the path is wrong".
const D = new URL("../../../demo/data", import.meta.url).pathname;
const URLS = {base:"x/",metadata:"x/m",hierarchy:"x/h",octree:"x/o"};

const HAS_DATA = existsSync(`${D}/large-100m/metadata.json`);
// Needs BOTH: the demo clouds, and a wasm built by a toolchain a clone may not
// have. See kernels.test.ts for the build line.
const WASM = new URL("../dist/voxelkloud_wasm_core.wasm", import.meta.url);
const CAN_RUN = HAS_DATA && existsSync(WASM);

/**
 * The measurement that justifies (or does not justify) this package.
 *
 * Skipped without demo/data, like the loader's drift guards. Warm the JIT and
 * take a median: an unwarmed batch swung 11% run to run and reported speedups
 * of 2.6-3.5x that do not survive warm-up.
 */
it.skipIf(!CAN_RUN)("wasm kernels against the TypeScript oracle", async () => {
  const bytes = readFileSync(WASM);
  const k = await loadWasmKernels(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset+bytes.byteLength));

  console.log("dataset       budget   nodes    points        TS ms   WASM ms   speedup  identical");
  for (const [dir, budgets] of [["real",[3e6,8e6]],["large-100m",[3e6,20e6,50e6]]] as const) {
    const src = parsePointCloudSource(JSON.parse(readFileSync(`${D}/${dir}/metadata.json`,"utf8")), URLS);
    const b = readFileSync(`${D}/${dir}/hierarchy.bin`);
    const h = createHierarchy(src, { buffer: b.buffer.slice(b.byteOffset, b.byteOffset+b.byteLength) });
    await h.expandAll();

    const t = src.tightBoundingBox;
    const cx=(t.min[0]+t.max[0])/2, cy=(t.min[1]+t.max[1])/2, cz=(t.min[2]+t.max[2])/2;
    const extent = Math.max(t.max[0]-t.min[0], t.max[1]-t.min[1], t.max[2]-t.min[2]);
    const dist = extent*0.25;
    // A hand-built clip matrix: this package must not depend on three.
    const f = 1/Math.tan((60*Math.PI/180)/2), near=Math.max(dist/500,.3), far=extent*20;
    const px=cx+dist*0.7, py=cy+dist*0.7, pz=cz+dist*0.45;
    const fx=cx-px, fy=cy-py, fz=cz-pz; const fl=Math.hypot(fx,fy,fz);
    const zx=-fx/fl, zy=-fy/fl, zz=-fz/fl;
    let ux=0,uy=0,uz=1;
    let sx=uy*zz-uz*zy, sy=uz*zx-ux*zz, sz=ux*zy-uy*zx; const sl=Math.hypot(sx,sy,sz);
    sx/=sl; sy/=sl; sz/=sl;
    const vx=zy*sz-zz*sy, vy=zz*sx-zx*sz, vz=zx*sy-zy*sx;
    const view=[sx,vx,zx,0, sy,vy,zy,0, sz,vz,zz,0,
      -(sx*px+sy*py+sz*pz), -(vx*px+vy*py+vz*pz), -(zx*px+zy*py+zz*pz), 1];
    const proj=[f/(16/9),0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,(2*far*near)/(near-far),0];
    const clip=new Float64Array(16);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s2=0;for(let i=0;i<4;i++)s2+=proj[i*4+r]!*view[c*4+i]!;clip[c*4+r]=s2;}

    for (const budget of budgets) {
      const sA=createLodScratch(), outA=createLodSelection(200_000);
      const sB=createLodScratch(), outB=createLodSelection(200_000);
      extractFrustumPlanes(clip, sA.planes, "minus-one-to-one", false);
      extractFrustumPlanes(clip, sB.planes, "minus-one-to-one", false);
      k.planes.set(sB.planes);
      const cam = {clipFromAbs:clip,camX:px,camY:py,camZ:pz,slope:Math.tan((60*Math.PI/180)/2),
        viewportHeightPx:2160,orthographic:false,orthoProjFactor:0,nearFloor:near,
        depthRange:"minus-one-to-one" as const,reversedDepth:false};
      const o = resolveLodOptions({targetScreenError:0.35,pointBudget:budget,maxNodes:200_000});

      selectVisible(h,cam,o,sA,outA);
      selectVisible(h,cam,o,sB,outB,k);
      const same = outA.count===outB.count && outA.points===outB.points &&
        Array.from(outA.indices.slice(0,outA.count)).every((v,i)=>v===outB.indices[i]);

      // Warm the JIT, then take the MEDIAN of repeated batches: a single batch
      // swung 11% between runs, which is more than some of the differences
      // being judged.
      const WARM=40, N=25, REPS=9;
      for(let i=0;i<WARM;i++){selectVisible(h,cam,o,sA,outA);selectVisible(h,cam,o,sB,outB,k);}
      // INTERLEAVED, and with the order alternating per rep.
      //
      // Measuring all nine TS batches and THEN all nine wasm batches puts the
      // two series in different thermal windows, and any drift across the ~450
      // selections between them is charged entirely to whichever ran second.
      // That is not a small effect at this granularity: with the sequential
      // form, one row read 0.93x, 1.19x, 1.12x and 1.10x across four
      // invocations of an unchanged binary. Interleaving makes both series see
      // the same conditions; alternating removes the residual bias of one
      // always being measured first within a rep.
      const batch=(f:()=>void)=>{const t0=performance.now();for(let i=0;i<N;i++)f();return (performance.now()-t0)/N;};
      const runTs=()=>batch(()=>selectVisible(h,cam,o,sA,outA));
      const runWs=()=>batch(()=>selectVisible(h,cam,o,sB,outB,k));
      const tsXs:number[]=[], wsXs:number[]=[];
      for(let r=0;r<REPS;r++){
        if(r%2===0){tsXs.push(runTs());wsXs.push(runWs());}
        else{wsXs.push(runWs());tsXs.push(runTs());}
      }
      const med=(xs:number[])=>{xs.sort((a,b)=>a-b); return xs[(xs.length-1)>>1]!;};
      const ts=med(tsXs), ws=med(wsXs);

      console.log(dir.padEnd(12), (budget/1e6+"M").padStart(7), String(outA.count).padStart(7),
        outA.points.toLocaleString().padStart(12), ts.toFixed(3).padStart(9), ws.toFixed(3).padStart(9),
        (ts/ws).toFixed(2)+"x", String(same).padStart(10));
    }
  }
}, 300000);
