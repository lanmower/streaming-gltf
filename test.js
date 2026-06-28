#!/usr/bin/env node
// Single real-services integration witness for the cluster-LOD pipeline.
// Mock-free: bakes a real corpus GLB through tools/bake-cluster.mjs, reads it back
// with real gltf-transform + meshoptimizer, and asserts the EP_cluster_lod format
// invariants the runtime depends on. No unit-mock harness; this proves a full bake
// -> valid-GLB -> parse round-trip end to end.
//
//   node test.js [path/to/source.glb]

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import draco3dgltf from 'draco3dgltf';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { rm } from 'node:fs/promises';
import { bakeCluster } from './tools/bake-cluster.mjs';
import { parseClusterLod, CLUSTER_LOD_EXTRA_KEY } from './examples/local-progressive/meshlet-codec.js';

const SRC = process.argv[2] || '../assets/Appliances/a_refrigerator_07685752_v1.glb';
const OUT = join(tmpdir(), `cluster-test-${process.pid}.glb`);

let failures = 0;
const ok = (cond, msg) => { console.log(`${cond ? '[pass]' : '[FAIL]'} ${msg}`); if (!cond) failures++; };

async function main() {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;

  // --- bake (real baker, real GLB) ---
  const res = await bakeCluster(SRC, OUT);
  ok(res.clustered >= 1, `baked >=1 clustered primitive (got ${res.clustered})`);
  ok(res.totalClusters >= 1, `produced clusters (got ${res.totalClusters})`);

  // --- read back with real gltf-transform + meshopt decode ---
  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
    'meshopt.encoder': MeshoptEncoder,
    'meshopt.decoder': MeshoptDecoder,
    'draco3d.decoder': await draco3dgltf.createDecoderModule(),
  });
  const doc = await io.read(OUT);
  const root = doc.getRoot();
  const accessors = root.listAccessors();

  const usedExts = root.listExtensionsUsed().map((e) => e.extensionName);
  ok(usedExts.includes('EXT_meshopt_compression'), `EXT_meshopt_compression present (${usedExts.join(',')})`);
  ok(!usedExts.includes('KHR_draco_mesh_compression'), 'stale KHR_draco_mesh_compression stripped (stock loader needs no DRACOLoader)');

  let clusterPrims = 0;
  for (const mesh of root.listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const extras = prim.getExtras() || {};
      if (!extras[CLUSTER_LOD_EXTRA_KEY]) continue;
      clusterPrims++;
      const meta = parseClusterLod(extras);
      ok(!!meta, 'parseClusterLod returned a ClusterSet');
      ok(meta.clusters.length === extras[CLUSTER_LOD_EXTRA_KEY].clusterCount, 'clusterCount matches clusters array');

      const idx = prim.getIndices();
      const lod0Count = idx.getCount();
      const coarseAcc = meta.coarseIndexAccessor >= 0 ? accessors[meta.coarseIndexAccessor] : null;
      const coarseCount = coarseAcc ? coarseAcc.getCount() : 0;
      ok(coarseCount === meta.coarseIndexCount, `coarse accessor count ${coarseCount} == meta ${meta.coarseIndexCount}`);

      // primitive.indices must equal LOD0 of every cluster = the full-res mesh a
      // stock viewer renders.
      let lod0sum = 0, badLod0 = 0, badCoarse = 0, wrongStream = 0, emptyLod = 0;
      for (const c of meta.clusters) {
        for (let l = 0; l < c.lods.length; l++) {
          const { offset, count, stream } = c.lods[l];
          if (count <= 0) emptyLod++;
          if (count % 3 !== 0) badLod0++;
          if (l === 0 && stream !== 0) wrongStream++;
          if (l > 0 && stream !== 1) wrongStream++;
          if (stream === 0) { lod0sum += count; if (offset + count > lod0Count) badLod0++; }
          else if (offset + count > coarseCount) badCoarse++;
        }
      }
      ok(lod0sum === lod0Count, `LOD0 ranges sum (${lod0sum}) == primitive.indices count (${lod0Count}) [stock full-res]`);
      ok(badLod0 === 0, 'all LOD0 ranges in-bounds & multiple-of-3');
      ok(badCoarse === 0, 'all coarse ranges in-bounds');
      ok(wrongStream === 0, 'lod0 -> stream 0, lod1+ -> stream 1');
      ok(emptyLod === 0, 'no empty/degenerate LOD range');

      // every index references a real vertex
      const vcount = prim.getAttribute('POSITION').getCount();
      let maxVid = 0; const arr = idx.getArray();
      for (let i = 0; i < arr.length; i++) if (arr[i] > maxVid) maxVid = arr[i];
      ok(maxVid < vcount, `max LOD0 vertex id ${maxVid} < vertexCount ${vcount}`);
    }
  }
  ok(clusterPrims >= 1, `found >=1 cluster primitive on read-back (${clusterPrims})`);

  await rm(OUT, { force: true });
  console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILURE(S)`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => { console.error('[FAIL] threw', e.message, e.stack); process.exit(1); });
