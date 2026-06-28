#!/usr/bin/env node
// Cluster-LOD baker (EP_cluster_lod).
//
// Reads a GLB and, for each UNSKINNED static primitive, rebuilds it as a single
// unified vertex+index buffer of UV-aware spatial meshlet clusters with per-cluster
// hierarchical LODs (see examples/local-progressive/meshlet-codec.js). Per-cluster
// AABB/sphere + per-(cluster,lod) index {offset,count} are written into
// primitive.extras.EP_cluster_lod (JSON only). The geometry stays a STANDARD single
// mesh/primitive: a stock glTF viewer ignores the extras and draws the whole index
// buffer = LOD0 of every cluster = the full-resolution mesh. EXT_meshopt_compression
// keeps the GLB small and valid.
//
// Skinned/morph primitives are left untouched (cluster-LOD is for static geometry;
// the runtime keeps its existing path for those).
//
// Run as a SEPARATE node process (heavy clustering OOMs an in-process host):
//   node tools/bake-cluster.mjs <input.glb> <output.glb>

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { dedup } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder, MeshoptSimplifier } from 'meshoptimizer';
import draco3dgltf from 'draco3dgltf';
import { buildClusterLod, buildClusterLodExtra, CLUSTER_LOD_EXTRA_KEY } from '../examples/local-progressive/meshlet-codec.js';
import { writeFile } from 'node:fs/promises';

// Map a gltf-transform primitive's accessors to the meshlet-codec geo shape.
// Attribute names are lowercased ('POSITION'->'position', 'TEXCOORD_0'->'texcoord_0')
// to match the codec's expectations; the codec keys position/uv off those names.
const ATTR_RENAME = { POSITION: 'position', NORMAL: 'normal', TANGENT: 'tangent', TEXCOORD_0: 'texcoord_0', COLOR_0: 'color' };

function primIsStatic(prim) {
  if (prim.getAttribute('JOINTS_0') || prim.getAttribute('WEIGHTS_0')) return false;
  if (prim.listTargets && prim.listTargets().length) return false;
  return true;
}

function primToGeo(prim) {
  const semantics = prim.listSemantics();
  const attributes = [];
  for (const sem of semantics) {
    const acc = prim.getAttribute(sem);
    if (!acc) continue;
    const name = ATTR_RENAME[sem] || sem.toLowerCase();
    attributes.push({ name, itemSize: acc.getElementSize(), normalized: acc.getNormalized(), array: acc.getArray(), _sem: sem });
  }
  const idxAcc = prim.getIndices();
  const index = idxAcc ? idxAcc.getArray() : null;
  return { attributes, index, _semByName: Object.fromEntries(attributes.map((a) => [a.name, a._sem])) };
}

async function bakeCluster(INPUT, OUTPUT) {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;
  await MeshoptSimplifier.ready;

  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
    'meshopt.encoder': MeshoptEncoder,
    'meshopt.decoder': MeshoptDecoder,
    'draco3d.decoder': await draco3dgltf.createDecoderModule(),
    'draco3d.encoder': await draco3dgltf.createEncoderModule(),
  });

  const doc = await io.read(INPUT);
  const root = doc.getRoot();
  const buffer = root.listBuffers()[0];

  let clustered = 0, skipped = 0, totalClusters = 0;
  const pendingExtras = []; // { prim, result, coarseAcc } resolved after transforms
  for (const mesh of root.listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      if (!primIsStatic(prim)) { skipped++; continue; }
      const geo = primToGeo(prim);
      if (!geo.attributes.find((a) => a.name === 'position')) { skipped++; continue; }

      const result = await buildClusterLod(geo, { maxVertices: 64, maxTriangles: 128, lodRatios: [1, 0.5, 0.25] });
      if (!result.clusters.length) { skipped++; continue; }

      // Rewrite attributes with the reordered unified arrays.
      for (const outAttr of result.attributes) {
        const sem = geo._semByName[outAttr.name];
        if (!sem) continue;
        const acc = doc
          .createAccessor()
          .setType(_glType(outAttr.itemSize))
          .setArray(outAttr.array)
          .setNormalized(outAttr.normalized)
          .setBuffer(buffer);
        prim.setAttribute(sem, acc);
      }
      // primitive.indices = LOD0 of every cluster = the full-resolution mesh, so
      // a stock glTF viewer that ignores extras draws the full mesh exactly once.
      const idxAcc = doc.createAccessor().setType('SCALAR').setArray(result.index).setBuffer(buffer);
      prim.setIndices(idxAcc);

      // Coarse (LOD1..N) indices live in a sidecar accessor referenced from extras.
      // A stock viewer never draws it; the runtime uses it for distant clusters.
      // gltf-transform's prune() would drop it (extras refs are invisible to the
      // graph), so we attach it to the prim's extension-less extras list and
      // resolve its FINAL accessor index after all transforms renumber accessors.
      let coarseAcc = null;
      if (result.indexCoarse.length) {
        coarseAcc = doc.createAccessor().setName('EP_cluster_lod_coarse').setType('SCALAR').setArray(result.indexCoarse).setBuffer(buffer);
      }
      pendingExtras.push({ prim, result, coarseAcc });

      clustered++;
      totalClusters += result.clusters.length;
    }
  }

  // Strip extensions the cluster GLB no longer uses. We re-encoded all geometry
  // with EXT_meshopt_compression, so KHR_draco_mesh_compression is dead; leaving
  // it in extensionsUsed forces stock GLTFLoader to demand a DRACOLoader (which it
  // throws without) even though no accessor is draco-compressed. EXT_texture_webp
  // stays — the textures are still webp.
  for (const ext of root.listExtensionsUsed()) {
    if (ext.extensionName === 'KHR_draco_mesh_compression') ext.dispose();
  }

  // dedup only. We deliberately AVOID the meshopt() transform: it runs reorder()
  // which re-permutes vertex/index buffers for GPU cache locality and would
  // DESTROY the cluster (offset,count) table the whole format depends on. The
  // codec already reordered vertices to index order, so reorder is redundant
  // anyway. We must also NOT prune(): it garbage-collects the coarse-index
  // accessors that only extras references.
  await doc.transform(dedup());

  // Compression is applied at WRITE time over all bufferViews (lossless FILTER
  // method = no vertex/index reorder, exact layout preserved), so the cluster
  // offsets stay valid. This keeps the GLB small + valid (EXT_meshopt_compression).
  const meshoptExt = doc.createExtension(EXTMeshoptCompression)
    .setRequired(true)
    .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.FILTER });

  // Now accessors are renumbered: resolve each coarse accessor's final index and
  // write the cluster extras (JSON only) onto its primitive.
  const finalAccessors = root.listAccessors();
  for (const { prim, result, coarseAcc } of pendingExtras) {
    const coarseAccessorIndex = coarseAcc ? finalAccessors.indexOf(coarseAcc) : -1;
    const extras = prim.getExtras() || {};
    extras[CLUSTER_LOD_EXTRA_KEY] = buildClusterLodExtra(result, coarseAccessorIndex);
    prim.setExtras(extras);
  }

  const bin = await io.writeBinary(doc);
  await writeFile(OUTPUT, Buffer.from(bin));
  console.log(`[bake-cluster] ${INPUT} -> ${OUTPUT}: clustered ${clustered} prim(s), ${totalClusters} clusters, skipped ${skipped} (skinned/morph), ${(bin.byteLength / 1024).toFixed(1)} KiB`);
  return { clustered, skipped, totalClusters, bytes: bin.byteLength };
}

function _glType(n) {
  return n === 1 ? 'SCALAR' : n === 2 ? 'VEC2' : n === 3 ? 'VEC3' : n === 4 ? 'VEC4' : 'SCALAR';
}

export { bakeCluster };

// CLI
if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith('bake-cluster.mjs')) {
  const [, , INPUT, OUTPUT] = process.argv;
  if (!INPUT || !OUTPUT) {
    console.error('usage: node tools/bake-cluster.mjs <input.glb> <output.glb>');
    process.exit(1);
  }
  bakeCluster(INPUT, OUTPUT).catch((e) => {
    console.error('[bake-cluster] ERROR', e.message, e.stack);
    process.exit(1);
  });
}
