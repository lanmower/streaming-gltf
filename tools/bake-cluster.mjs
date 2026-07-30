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
import { dedup, simplify, cloneDocument } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder, MeshoptSimplifier } from 'meshoptimizer';
import draco3dgltf from 'draco3dgltf';
import { buildClusterLod, buildClusterLodExtra, CLUSTER_LOD_EXTRA_KEY } from '../examples/local-progressive/meshlet-codec.js';
import { writeFile, mkdir, readFile, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';

// Discrete-LOD ratios for SKINNED/morph primitives (cluster-LOD cannot handle them
// -- it needs static topology). meshopt simplify() preserves JOINTS_0/WEIGHTS_0 +
// morph deltas (the simplified index is a subset of original vertices), so a skinned
// VRM gets real LOD scaling. Lowest detail first matches the runtime sort (ascending
// quality). 1.0 is the inline base in the root; the rest are sibling files.
const SKINNED_LOD_RATIOS = [1.0, 0.4, 0.15];
const EP_PROGRESSIVE_LOD_KEY = 'EP_progressive_lod';

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

// Build discrete LOD siblings for ONE skinned primitive. Clones the document down
// to just this primitive, meshopt-simplifies it per ratio (preserving skin attrs +
// morphs), and writes each LOD<1.0 as a standalone sibling GLB under <outDir>/lods/.
// Returns { meshIndex, primIndex, lods:[...] } for the EP_progressive_lod payload,
// where exactly one entry (ratio 1.0) is inline:true (drawn from the root). The
// runtime (model-pool.js _applyLod skinned branch) swaps the sibling geometry onto
// the root's shared skeleton, so the sibling needs no skeleton of its own -- only
// JOINTS_0/WEIGHTS_0 that index the same joints, which simplify() preserves.
async function _bakeSkinnedLods(srcDoc, io, meshIndex, primIndex, lodsDir, baseName) {
  const lods = [];
  for (const ratio of SKINNED_LOD_RATIOS) {
    if (ratio >= 1.0) { lods.push({ ratio: 1.0, kind: 'textured', inline: true }); continue; }
    // Fresh clone per ratio so each simplify starts from the full-res source
    // (simplify is destructive; chaining ratios would compound error). Cloned
    // in-memory from the already-parsed source document instead of re-reading
    // + re-parsing the GLB off disk for every ratio (was 3x redundant I/O+parse
    // per skinned primitive; cloneDocument gives an equally-fresh independent
    // Document via gltf-transform's own deep merge).
    const doc = cloneDocument(srcDoc);
    const root = doc.getRoot();
    const meshes = root.listMeshes();
    const mesh = meshes[meshIndex];
    if (!mesh) break;
    const prims = mesh.listPrimitives();
    const keepPrim = prims[primIndex];
    if (!keepPrim) break;
    // Strip every OTHER mesh + every other primitive so the sibling is geometry-only,
    // single-primitive (the worker takes the first mesh it finds).
    for (const m of meshes) {
      for (const p of m.listPrimitives()) { if (p !== keepPrim) m.removePrimitive(p); }
      if (m !== mesh) m.dispose();
    }
    const pos = keepPrim.getAttribute('POSITION');
    if (!pos) break;
    // decodeAABB = POSITION min/max BEFORE meshopt quantization (the worker rescales
    // the decoded [-1,1]-ish positions back into character-local space with this).
    const min = pos.getMinNormalized ? pos.getMin([]) : pos.getMin([]);
    const max = pos.getMax([]);
    const decodeAABB = { min: [min[0], min[1], min[2]], max: [max[0], max[1], max[2]] };
    try {
      await doc.transform(simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.01, lockBorder: false }));
    } catch (e) { continue; }
    const idxAcc = keepPrim.getIndices();
    const vCount = keepPrim.getAttribute('POSITION')?.getCount() || 0;
    const iCount = idxAcc ? idxAcc.getCount() : 0;
    if (iCount === 0 || vCount === 0) continue;   // simplified to a hole -> skip
    // meshopt-encode the sibling at write time.
    doc.createExtension(EXTMeshoptCompression)
      .setRequired(true)
      .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.FILTER });
    const bin = await io.writeBinary(doc);
    const fileName = `${baseName}_m${meshIndex}_p${primIndex}_r${String(ratio).replace('.', '')}.glb`;
    await mkdir(lodsDir, { recursive: true });
    await writeFile(join(lodsDir, fileName), Buffer.from(bin));
    lods.push({ ratio, kind: 'textured', path: `lods/${fileName}`, inline: false, indexCount: iCount, vertexCount: vCount, bytes: bin.byteLength, decodeAABB });
  }
  // Only worth a descriptor if at least one real sibling LOD was emitted.
  const siblingCount = lods.filter((l) => !l.inline).length;
  if (siblingCount === 0) return null;
  return { meshIndex, primIndex, lods };
}

// Pre-flight validation of INPUT: a missing file or a file that isn't actually
// a GLB previously fell straight into io.read(INPUT), which throws an opaque
// gltf-transform-internal error with no hint the real problem was "wrong path"
// or "not a GLB" -- surfaces a clear, actionable message instead. The CLI entry
// point already wraps bakeCluster() in a .catch, but that only helps when
// invoked from the command line; a programmatic caller (e.g. bake-cluster-corpus.mjs,
// or a consumer importing { bakeCluster } directly) gets the same opaque error
// without this check.
async function _validateInputGlb(INPUT) {
  let st;
  try {
    st = await stat(INPUT);
  } catch (e) {
    throw new Error(`bakeCluster: INPUT not found or unreadable: ${INPUT} (${e.code || e.message})`);
  }
  if (!st.isFile()) throw new Error(`bakeCluster: INPUT is not a file: ${INPUT}`);
  const fh = await readFile(INPUT);
  if (fh.byteLength < 4 || fh.readUInt32LE(0) !== 0x46546c67) {
    throw new Error(`bakeCluster: INPUT is not a valid GLB (bad magic): ${INPUT}`);
  }
}

async function bakeCluster(INPUT, OUTPUT) {
  await _validateInputGlb(INPUT);
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
  // A degenerate/malformed glTF with zero buffers would otherwise let every
  // later `.setBuffer(buffer)` silently attach an accessor to `undefined`,
  // producing a corrupt output GLB instead of a clear upfront failure. Only
  // an actual clustering candidate needs a buffer to write into, so this check
  // fires lazily -- right before the first prim that would need one -- rather
  // than unconditionally (a document with only skinned/skipped prims and no
  // static geometry to cluster never needs to write a new accessor at all).

  let clustered = 0, skipped = 0, totalClusters = 0, skinnedLodded = 0;
  const pendingExtras = []; // { prim, result, coarseAcc } resolved after transforms
  const skinnedDescs = []; // EP_progressive_lod mesh descriptors (skinned discrete LODs)
  const lodsDir = join(dirname(OUTPUT), 'lods');
  const baseName = 'sk';
  const allMeshes = root.listMeshes();
  for (let mi = 0; mi < allMeshes.length; mi++) {
    const mesh = allMeshes[mi];
    const prims = mesh.listPrimitives();
    for (let pi = 0; pi < prims.length; pi++) {
      const prim = prims[pi];
      if (!primIsStatic(prim)) {
        // Skinned/morph: cluster-LOD can't handle it, but we still give it discrete
        // meshopt LODs (sibling GLBs + EP_progressive_lod) so a VRM/skinned model gets
        // real LOD scaling through ModelPool's skinned LOD ladder.
        try {
          const desc = await _bakeSkinnedLods(doc, io, mi, pi, lodsDir, baseName);
          if (desc) { skinnedDescs.push(desc); skinnedLodded++; }
          else skipped++;
        } catch (e) { console.warn(`[bake-cluster] skinned LOD skipped (mesh ${mi} prim ${pi}): ${e.message}`); skipped++; }
        continue;
      }
      const geo = primToGeo(prim);
      if (!geo.attributes.find((a) => a.name === 'position')) { skipped++; continue; }

      const result = await buildClusterLod(geo, { maxVertices: 64, maxTriangles: 128, lodRatios: [1, 0.5, 0.25] });
      if (!result.clusters.length) { skipped++; continue; }
      if (!buffer) throw new Error(`bakeCluster: document has a clusterable static primitive (mesh ${mi} prim ${pi}) but no buffer to write the reordered accessors into (root.listBuffers() is empty) -- malformed glTF`);

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

  let bin = await io.writeBinary(doc);

  // Splice the EP_progressive_lod payload (skinned discrete LODs) into the root GLB
  // JSON chunk. gltf-transform drops unknown top-level extensions on write, so we
  // rewrite the JSON chunk by hand. The skinned full-res mesh is already INLINE in
  // the root (we never removed it), so each descriptor's inline:true LOD draws from
  // the root primitive; the sibling LODs live under lods/ and are fetched on demand.
  if (skinnedDescs.length) {
    bin = _spliceProgressiveLod(bin, skinnedDescs);
  }

  await writeFile(OUTPUT, Buffer.from(bin));
  console.log(`[bake-cluster] ${INPUT} -> ${OUTPUT}: clustered ${clustered} prim(s), ${totalClusters} clusters, skinned-lodded ${skinnedLodded} prim(s), skipped ${skipped}, ${(bin.byteLength / 1024).toFixed(1)} KiB`);
  return { clustered, skipped, totalClusters, skinnedLodded, bytes: bin.byteLength };
}

// Rewrite a GLB's JSON chunk to carry extensions.EP_progressive_lod (+ list it in
// extensionsUsed, never extensionsRequired so a stock viewer still draws the inline
// base). The BIN chunk is copied through untouched; only the JSON chunk grows.
function _spliceProgressiveLod(bin, meshes) {
  const u8 = bin instanceof Uint8Array ? bin : new Uint8Array(bin);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) return bin; // not a GLB
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(u8.subarray(20, 20 + jsonLen)));
  json.extensions = json.extensions || {};
  json.extensions[EP_PROGRESSIVE_LOD_KEY] = { version: 1, storage: 'sibling-file', meshes };
  const used = new Set(json.extensionsUsed || []);
  used.add(EP_PROGRESSIVE_LOD_KEY);
  json.extensionsUsed = [...used];
  let nj = JSON.stringify(json);
  while (nj.length % 4 !== 0) nj += ' ';
  const jb = new TextEncoder().encode(nj);
  const binChunkStart = 20 + jsonLen;
  const binChunkLen = dv.getUint32(binChunkStart, true);
  const binChunkType = dv.getUint32(binChunkStart + 4, true);
  const binData = u8.subarray(binChunkStart + 8, binChunkStart + 8 + binChunkLen);
  const total = 12 + 8 + jb.length + 8 + binData.length;
  const out = new Uint8Array(total);
  const odv = new DataView(out.buffer);
  odv.setUint32(0, 0x46546c67, true); odv.setUint32(4, 2, true); odv.setUint32(8, total, true);
  odv.setUint32(12, jb.length, true); odv.setUint32(16, 0x4e4f534a, true); out.set(jb, 20);
  let o = 20 + jb.length;
  odv.setUint32(o, binData.length, true); odv.setUint32(o + 4, binChunkType, true); out.set(binData, o + 8);
  return out;
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
