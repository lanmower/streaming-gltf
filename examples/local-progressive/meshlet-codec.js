// Meshlet cluster-LOD codec ("EP_cluster_lod").
//
// UV-Aware Spatial Clustering with Hierarchical Index Packing. The whole mesh is
// divided into spatially coherent meshlets (clusters of <=maxTris triangles); for
// each cluster a hierarchy of UV-aware simplified LODs is generated; all clusters
// and their LODs are packed into ONE unified, reordered vertex buffer + one index
// buffer ordered [cluster0:lod0,lod1,..][cluster1:lod0,..]... so a GPU fetching a
// cluster's chosen LOD pulls a tiny contiguous block. Per (cluster,lod) we record
// {offset,count} into the index buffer plus a per-cluster AABB + bounding sphere,
// so the renderer can frustum-cull clusters and pick a LOD per cluster, then issue
// the visible index sub-ranges via WEBGL_multi_draw in a single draw call.
//
// The unified buffer is a STANDARD vertex/index buffer: a stock glTF viewer that
// ignores extras.EP_cluster_lod just draws the whole index buffer = LOD0 of every
// cluster = the full-resolution mesh. The cluster metadata lives only in extras
// (JSON) so the GLB stays 100% valid.
//
// BUILDER (buildClusterLod) lazily imports meshoptimizer and runs at bake time.
// The metadata SCHEMA + the reader (parseClusterLod) are dependency-free and run
// in the browser.

export const CLUSTER_LOD_VERSION = 1;
export const CLUSTER_LOD_EXTRA_KEY = 'EP_cluster_lod';

// ---------------------------------------------------------------- builder ----

let _meshopt = null;
async function _ensureMeshopt() {
  if (_meshopt) return _meshopt;
  const { MeshoptClusterizer, MeshoptSimplifier } = await import('meshoptimizer');
  await MeshoptClusterizer.ready;
  await MeshoptSimplifier.ready;
  _meshopt = { MeshoptClusterizer, MeshoptSimplifier };
  return _meshopt;
}

// Synthesize a sequential index for non-indexed triangle soup (some primitives
// arrive non-indexed). A raw sequential index has no shared edges so simplify
// cannot reduce it; callers that need real decimation should weld first, but
// clustering + LOD0 still work on the soup.
function _seqIndex(vertCount) {
  const ix = new Uint32Array(vertCount);
  for (let i = 0; i < vertCount; i++) ix[i] = i;
  return ix;
}

// One attribute stream description: {name, array(Float32Array|TypedArray),
// itemSize, normalized}. POSITION is required and must be name 'position'.
//
// opts:
//   maxVertices (64), maxTriangles (128), minTriangles (max(1,maxTriangles>>2)),
//   coneOrFillWeight (0.5),
//   lodRatios ([1, 0.5, 0.25]) - target index fraction per LOD relative to the
//     cluster's LOD0 triangle count,
//   lodError (0.05) - relative target error passed to simplify,
//   uvAware (true) - weight TEXCOORD_0 in the QEM so UV seams are preserved.
//
// Returns:
//   {
//     vertexCount, attributes: [{name,itemSize,normalized, array}],  // unified, reordered
//     index: Uint32Array,                                            // unified
//     clusters: [{ aabb:[minx,miny,minz,maxx,maxy,maxz],
//                  sphere:[cx,cy,cz,r],
//                  lods:[{offset,count}] }],   // offset/count into `index`
//     lodCount,
//   }
export async function buildClusterLod(geo, opts = {}) {
  const { MeshoptClusterizer: C, MeshoptSimplifier: S } = await _ensureMeshopt();

  const maxVertices = opts.maxVertices || 64;
  const maxTriangles = opts.maxTriangles || 128;
  const minTriangles = opts.minTriangles || Math.max(1, maxTriangles >> 2);
  const fillWeight = opts.coneOrFillWeight != null ? opts.coneOrFillWeight : 0.5;
  const lodRatios = opts.lodRatios || [1, 0.5, 0.25];
  const lodError = opts.lodError != null ? opts.lodError : 0.05;
  const uvAware = opts.uvAware !== false;

  const attrs = geo.attributes;
  const posAttr = attrs.find((a) => a.name === 'position');
  if (!posAttr) throw new Error('buildClusterLod: missing position attribute');
  const srcVertCount = posAttr.array.length / posAttr.itemSize;

  // meshopt wants tightly packed Float32 xyz with stride in FLOATS.
  const position =
    posAttr.array instanceof Float32Array && posAttr.itemSize === 3
      ? posAttr.array
      : _toFloat32Vec3(posAttr.array, posAttr.itemSize);

  const index = geo.index ? _toUint32(geo.index) : _seqIndex(srcVertCount);

  // UV stream for attribute-aware simplify (preserves texture mapping).
  const uvAttr = uvAware ? attrs.find((a) => a.name === 'uv' || a.name === 'texcoord_0') : null;
  const uvArr = uvAttr ? _toFloat32(uvAttr.array, uvAttr.itemSize, 2) : null;

  // 1. Spatial clustering into meshlets.
  const mb = C.buildMeshletsSpatial(
    index,
    position,
    3,
    maxVertices,
    minTriangles,
    maxTriangles,
    fillWeight
  );
  const bounds = C.computeMeshletBounds(mb, position, 3);

  // 2..3. Per cluster: LOD0 = the cluster's own triangles (global vertex ids);
  // LOD1..N = UV-aware simplifications. Pack everything into a fresh, reordered
  // unified vertex buffer (vertices appended in first-seen order across the
  // packed index stream -> contiguous per cluster for GPU fetch locality).
  const lodCount = lodRatios.length;
  const remap = new Int32Array(srcVertCount).fill(-1); // old vid -> new vid
  const newOrder = []; // new vid -> old vid
  // TWO index streams so a stock viewer drawing primitive.indices renders exactly
  // the full-res mesh: index0 = LOD0 of every cluster (-> primitive.indices);
  // indexCoarse = LOD1..N (-> a sidecar accessor referenced from extras). A
  // cluster's lods[0].offset/count index into index0; lods[1..] into indexCoarse.
  // `stream:0|1` tags which buffer the range lives in.
  const index0 = [];
  const indexCoarse = [];
  const clusters = [];

  // Append `glob` (global vertex ids) to `stream`, remapping to the shared vertex
  // table, and return {offset, count, stream:streamTag}.
  const appendTo = (stream, streamTag, glob) => {
    const offset = stream.length;
    for (let i = 0; i < glob.length; i++) {
      const old = glob[i];
      let nv = remap[old];
      if (nv === -1) {
        nv = newOrder.length;
        remap[old] = nv;
        newOrder.push(old);
      }
      stream.push(nv);
    }
    return { offset, count: glob.length, stream: streamTag };
  };

  const uvW = uvArr ? _uvWeights(uvAttr) : null;
  for (let m = 0; m < mb.meshletCount; m++) {
    const mesh = C.extractMeshlet(mb, m);
    const clusterVerts = mesh.vertices; // global vertex ids in this meshlet (<=maxVertices)

    // Build a COMPACT per-cluster vertex table (positions + uvs) so simplify runs
    // against tiny arrays, not the full-mesh arrays. Passing the whole 500k-vertex
    // position array per cluster per LOD is O(clusters * lods * totalVerts) and
    // pathologically slow on large meshes — local tables make each simplify O(64).
    const lv = clusterVerts.length;
    const localPos = new Float32Array(lv * 3);
    const localUv = uvArr ? new Float32Array(lv * 2) : null;
    for (let i = 0; i < lv; i++) {
      const g = clusterVerts[i];
      localPos[i * 3] = position[g * 3]; localPos[i * 3 + 1] = position[g * 3 + 1]; localPos[i * 3 + 2] = position[g * 3 + 2];
      if (localUv) { localUv[i * 2] = uvArr[g * 2]; localUv[i * 2 + 1] = uvArr[g * 2 + 1]; }
    }
    // LOD0 as LOCAL indices (into clusterVerts); mesh.triangles are already local.
    const lod0Local = mesh.triangles; // Uint8Array of local vertex ids
    const lod0LocalU32 = new Uint32Array(lod0Local.length);
    for (let i = 0; i < lod0Local.length; i++) lod0LocalU32[i] = lod0Local[i];

    const lods = [];
    let prevLocal = lod0LocalU32;
    for (let l = 0; l < lodCount; l++) {
      let local;
      if (l === 0) {
        local = lod0LocalU32;
      } else {
        const tRaw = Math.round(lod0LocalU32.length * lodRatios[l]);
        const targetIdx = Math.max(3, tRaw - (tRaw % 3));
        if (targetIdx >= prevLocal.length) {
          local = prevLocal;
        } else {
          const [si] = localUv
            ? S.simplifyWithAttributes(prevLocal, localPos, 3, localUv, 2, uvW, null, targetIdx, lodError, ['LockBorder'])
            : S.simplify(prevLocal, localPos, 3, targetIdx, lodError, ['LockBorder']);
          local = si.length >= 3 ? si : prevLocal;
        }
      }
      prevLocal = local;
      // map local ids -> global ids for the shared-table append
      const glob = new Uint32Array(local.length);
      for (let i = 0; i < local.length; i++) glob[i] = clusterVerts[local[i]];
      lods.push(l === 0 ? appendTo(index0, 0, glob) : appendTo(indexCoarse, 1, glob));
    }
    const lod0 = new Uint32Array(lod0Local.length);
    for (let i = 0; i < lod0Local.length; i++) lod0[i] = clusterVerts[lod0Local[i]];

    const b = bounds[m];
    // AABB from the cluster's LOD0 vertices.
    let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
    for (let i = 0; i < lod0.length; i++) {
      const v = lod0[i];
      const x = position[v * 3], y = position[v * 3 + 1], z = position[v * 3 + 2];
      if (x < mnx) mnx = x; if (x > mxx) mxx = x;
      if (y < mny) mny = y; if (y > mxy) mxy = y;
      if (z < mnz) mnz = z; if (z > mxz) mxz = z;
    }
    clusters.push({
      aabb: [mnx, mny, mnz, mxx, mxy, mxz],
      sphere: [b.centerX, b.centerY, b.centerZ, b.radius],
      lods,
    });
  }

  // 4. Build the reordered unified attribute buffers in newOrder.
  const newVertCount = newOrder.length;
  const outAttrs = attrs.map((a) => {
    const Ctor = a.array.constructor;
    const out = new Ctor(newVertCount * a.itemSize);
    for (let nv = 0; nv < newVertCount; nv++) {
      const old = newOrder[nv];
      for (let c = 0; c < a.itemSize; c++) out[nv * a.itemSize + c] = a.array[old * a.itemSize + c];
    }
    return { name: a.name, itemSize: a.itemSize, normalized: !!a.normalized, array: out };
  });

  return {
    vertexCount: newVertCount,
    attributes: outAttrs,
    index: Uint32Array.from(index0), // LOD0 of all clusters -> primitive.indices (stock full-res draw)
    indexCoarse: Uint32Array.from(indexCoarse), // LOD1..N -> sidecar accessor referenced from extras
    clusters,
    lodCount,
  };
}

// UV weights: heavier weight = stronger penalty on collapsing edges that distort
// UVs, which keeps texture mapping intact (the "UV-aware" penalty).
function _uvWeights(uvAttr) {
  const n = uvAttr ? Math.min(uvAttr.itemSize, 2) : 2;
  const w = [];
  for (let i = 0; i < n; i++) w.push(0.5);
  return w;
}

function _toUint32(arr) {
  return arr instanceof Uint32Array ? arr : Uint32Array.from(arr);
}
function _toFloat32Vec3(arr, itemSize) {
  const n = arr.length / itemSize;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = arr[i * itemSize];
    out[i * 3 + 1] = arr[i * itemSize + 1];
    out[i * 3 + 2] = arr[i * itemSize + 2];
  }
  return out;
}
function _toFloat32(arr, itemSize, want) {
  if (arr instanceof Float32Array && itemSize === want) return arr;
  const n = arr.length / itemSize;
  const out = new Float32Array(n * want);
  for (let i = 0; i < n; i++) for (let c = 0; c < want; c++) out[i * want + c] = arr[i * itemSize + c] || 0;
  return out;
}

// --------------------------------------------------------- extras schema ----

// Build the JSON metadata object stored at primitive.extras[CLUSTER_LOD_EXTRA_KEY].
// Stock viewers ignore extras and draw the full index buffer (= LOD0 of all
// clusters = full mesh). For meshes with very many clusters the caller may instead
// store the bounds/offset/count arrays as a packed glTF accessor and reference it
// here (packed:true); the inline form below is the simple JSON path.
// `coarseIndexAccessor` is the glTF accessor index for the LOD1..N index buffer
// (set by the baker after it creates that accessor). lods entries are
// [offset, count, stream] where stream 0 = primitive.indices (LOD0), 1 = the
// coarse accessor.
export function buildClusterLodExtra(result, coarseIndexAccessor = -1) {
  return {
    version: CLUSTER_LOD_VERSION,
    clusterCount: result.clusters.length,
    lodCount: result.lodCount,
    coarseIndexAccessor,
    coarseIndexCount: result.indexCoarse.length,
    clusters: result.clusters.map((c) => ({
      aabb: c.aabb.map(_round),
      sphere: c.sphere.map(_round),
      lods: c.lods.map((l) => [l.offset, l.count, l.stream]),
    })),
  };
}
function _round(v) {
  return Math.round(v * 1e4) / 1e4;
}

// ----------------------------------------------------------------- reader ----

// Parse extras.EP_cluster_lod into a normalized ClusterSet. Dependency-free,
// browser-safe. Returns null when the extra is absent/invalid (caller falls back
// to full-draw = stock behavior).
export function parseClusterLod(extras) {
  const meta = extras && extras[CLUSTER_LOD_EXTRA_KEY];
  if (!meta || !Array.isArray(meta.clusters) || !meta.clusters.length) return null;
  const clusters = meta.clusters.map((c) => ({
    aabb: c.aabb,
    sphere: c.sphere,
    lods: c.lods.map((l) =>
      Array.isArray(l)
        ? { offset: l[0], count: l[1], stream: l[2] || 0 }
        : { offset: l.offset, count: l.count, stream: l.stream || 0 }
    ),
  }));
  return {
    version: meta.version || 1,
    lodCount: meta.lodCount || (clusters[0] && clusters[0].lods.length) || 1,
    coarseIndexAccessor: meta.coarseIndexAccessor != null ? meta.coarseIndexAccessor : -1,
    coarseIndexCount: meta.coarseIndexCount || 0,
    clusters,
  };
}
