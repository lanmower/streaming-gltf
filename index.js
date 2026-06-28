// streaming-gltf — public SDK entry.
//
// Cluster-LOD glTF renderer: each asset is baked into UV-aware spatial meshlet
// clusters with per-cluster hierarchical LODs packed into one unified buffer
// (EP_cluster_lod extras + EXT_meshopt_compression, a 100%-valid single GLB). At
// runtime each visible cluster picks a LOD by projected size and the whole mesh
// draws in a single WEBGL_multi_draw call. Stock glTF viewers ignore the extras
// and render the full-resolution mesh.
//
// `three` is a peer dependency — provide it yourself (e.g. via an importmap
// pointing at a CDN build, or your bundler). This package does not bundle three.
//
//   import { ModelPool } from 'streaming-gltf';
//   const pool = new ModelPool({ scene, renderer, camera });
//   const e = pool.spawn(url, { position: [x, 0, z] });
//   pool.update(); // per-frame, after advancing the camera
//   pool.setTarget(e, x, y, z, durationMs); // GPU-interpolated position targets
//
// Bake a GLB to the cluster-LOD format with `npm run bake -- <in.glb> <out.glb>`
// (tools/bake-cluster.mjs) or a whole corpus with `npm run bake:corpus`.

export { ModelPool } from './examples/local-progressive/model-pool.js';
export { ClusterLodMesh, attachClusterLod } from './examples/local-progressive/cluster-lod-mesh.js';
export { buildClusterLod, buildClusterLodExtra, parseClusterLod, CLUSTER_LOD_EXTRA_KEY } from './examples/local-progressive/meshlet-codec.js';
