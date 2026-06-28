// Runtime cluster-LOD mesh (EP_cluster_lod consumer).
//
// Wraps a single unified geometry (one mesh / one primitive, the full-res LOD0 in
// geometry.index + the coarse LOD1..N indices appended) plus the per-cluster
// metadata parsed from extras.EP_cluster_lod. Each frame it:
//   1. frustum-culls clusters by their bounding sphere (CPU),
//   2. picks a LOD per visible cluster from projected screen size,
//   3. accumulates the chosen index sub-ranges, and
//   4. issues them via WEBGL/ANGLE_multi_draw in a SINGLE draw call against the
//      unified element buffer (falls back to a per-range drawElements loop, or to
//      a plain full-LOD0 draw, when the extension is absent).
//
// The whole index buffer = LOD0 of every cluster, so if this object is ever drawn
// by the stock three pipeline (no onBeforeRender override applied) it still renders
// the correct full-resolution mesh.

import * as THREE from 'three';
import { parseClusterLod } from './meshlet-codec.js';

const _sphere = new THREE.Sphere();
const _frustum = new THREE.Frustum();
const _projScreen = new THREE.Matrix4();
const _v = new THREE.Vector3();
const _size = new THREE.Vector2();

// thresholds: projected sphere radius (px-ish, screenH * r / dist) above which a
// given LOD is used. Index i is chosen when projected size > thresholds[i].
// Descending: big on screen -> LOD0, small -> coarsest.
const DEFAULT_LOD_THRESHOLDS = [120, 40]; // LOD0 if >120, LOD1 if >40, else LOD2

export class ClusterLodMesh extends THREE.Mesh {
  // geometry: BufferGeometry whose .index already contains [LOD0 ... | coarse ...]
  //   concatenated (lod0Count = number of LOD0 indices; coarse indices follow).
  // clusterSet: output of parseClusterLod(extras) with cluster.lods[].stream/offset/count.
  // opts: { lodThresholds, screenHeight, hysteresis }
  constructor(geometry, material, clusterSet, opts = {}) {
    super(geometry, material);
    this.clusterSet = clusterSet;
    this.lod0Count = opts.lod0Count != null ? opts.lod0Count : _inferLod0Count(clusterSet);
    this.lodThresholds = opts.lodThresholds || DEFAULT_LOD_THRESHOLDS;
    this._screenHeight = opts.screenHeight || 1080;
    this._hyst = opts.hysteresis != null ? opts.hysteresis : 0.15;
    this._curLod = new Int8Array(clusterSet.clusters.length).fill(-1);

    // Per-frame scratch (sized to worst case = every cluster drawn).
    const n = clusterSet.clusters.length;
    this._starts = new Int32Array(n); // byte offsets into element buffer
    this._counts = new Int32Array(n);
    this._drawCount = 0;

    this._ext = null;
    this._extProbed = false;

    // Live stats for the browser witness.
    this.stats = { visibleClusters: 0, drawnTris: 0, totalTris: 0, multiDrawSubmissions: 0, ext: null };
    for (const c of clusterSet.clusters) this.stats.totalTris += c.lods[0].count / 3;

    // Take over drawing.
    this.onBeforeRender = this._render.bind(this);
    this.frustumCulled = false; // we cull per-cluster ourselves
  }

  // Map a cluster lod descriptor to a byte offset into the unified element buffer.
  // stream 0 = LOD0 region (offset as-is); stream 1 = coarse region (after lod0Count).
  _byteOffset(lod, bytesPerIndex) {
    const base = lod.stream === 1 ? this.lod0Count : 0;
    return (base + lod.offset) * bytesPerIndex;
  }

  _pickLod(ci, projSize) {
    const t = this.lodThresholds;
    const cur = this._curLod[ci];
    let lod = t.length; // default coarsest
    for (let i = 0; i < t.length; i++) {
      // hysteresis: to gain detail (lower i) require clearing threshold by +margin;
      // to drop require falling below by -margin. Bias by current level.
      const goingUp = cur < 0 || cur > i;
      const eff = goingUp ? t[i] * (1 + this._hyst) : t[i] * (1 - this._hyst);
      if (projSize > eff) { lod = i; break; }
    }
    // clamp to available LODs for this cluster
    const avail = this.clusterSet.clusters[ci].lods.length;
    if (lod >= avail) lod = avail - 1;
    this._curLod[ci] = lod;
    return lod;
  }

  _render(renderer, scene, camera, geometry) {
    if (!this._extProbed) {
      const gl = renderer.getContext();
      this._ext = gl.getExtension('WEBGL_multi_draw') || gl.getExtension('ANGLE_multi_draw') || null;
      this.stats.ext = this._ext ? 'WEBGL_multi_draw' : 'none';
      this._extProbed = true;
    }

    const index = geometry.index;
    if (!index || !this.clusterSet) return; // nothing to do; default draw renders full LOD0
    const bytesPerIndex = index.array.BYTES_PER_ELEMENT;

    _projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_projScreen);
    const camPos = camera.getWorldPosition(_v.copy(camera.position));
    // Live viewport height from the renderer's drawing buffer (falls back to the
    // constructor value) so projected-size LOD thresholds track the real canvas.
    let sh = this._screenHeight;
    try { const sz = renderer.getDrawingBufferSize(_size); if (sz.y > 0) sh = sz.y; } catch (_) {}
    // fov-based projection scale (perspective): projSize ~= sh * r / (dist * tan(fov/2)).
    const tanHalf = camera.isPerspectiveCamera ? Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5) : 1;
    const me = this.matrixWorld.elements;
    const scale = Math.max(
      Math.hypot(me[0], me[1], me[2]),
      Math.hypot(me[4], me[5], me[6]),
      Math.hypot(me[8], me[9], me[10])
    );

    let dc = 0, drawnTris = 0, visible = 0;
    const clusters = this.clusterSet.clusters;
    for (let ci = 0; ci < clusters.length; ci++) {
      const c = clusters[ci];
      // world-space sphere
      _sphere.center.set(c.sphere[0], c.sphere[1], c.sphere[2]).applyMatrix4(this.matrixWorld);
      _sphere.radius = c.sphere[3] * scale;
      if (!_frustum.intersectsSphere(_sphere)) continue;
      visible++;
      const dist = Math.max(1e-3, _sphere.center.distanceTo(camPos));
      const projSize = (sh * _sphere.radius) / (dist * tanHalf);
      const lodIdx = this._pickLod(ci, projSize);
      const lod = c.lods[lodIdx];
      if (!lod.count) continue;
      this._starts[dc] = this._byteOffset(lod, bytesPerIndex);
      this._counts[dc] = lod.count;
      drawnTris += lod.count / 3;
      dc++;
    }
    this._drawCount = dc;
    this.stats.visibleClusters = visible;
    this.stats.drawnTris = drawnTris;

    this._draw(renderer, geometry, index);
  }

  _draw(renderer, geometry, index) {
    const gl = renderer.getContext();
    const dc = this._drawCount;
    if (dc === 0) { this.stats.multiDrawSubmissions = 0; return; }

    // Ensure the element buffer + VAO state is bound by letting three set up the
    // program/attributes for this mesh, then override the draw. We bind the
    // geometry's element buffer explicitly.
    const glType = index.array.BYTES_PER_ELEMENT === 2 ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;

    if (this._ext && this._ext.multiDrawElementsWEBGL) {
      this._ext.multiDrawElementsWEBGL(gl.TRIANGLES, this._counts, 0, glType, this._starts, 0, dc);
      this.stats.multiDrawSubmissions = 1;
    } else if (this._ext && this._ext.multiDrawElementsANGLE) {
      this._ext.multiDrawElementsANGLE(gl.TRIANGLES, this._counts, 0, glType, this._starts, 0, dc);
      this.stats.multiDrawSubmissions = 1;
    } else {
      // Fallback: per-range drawElements loop.
      for (let i = 0; i < dc; i++) gl.drawElements(gl.TRIANGLES, this._counts[i], glType, this._starts[i]);
      this.stats.multiDrawSubmissions = dc;
    }
  }
}

function _inferLod0Count(clusterSet) {
  let n = 0;
  for (const c of clusterSet.clusters) {
    const l0 = c.lods[0];
    if (l0.stream === 0) n = Math.max(n, l0.offset + l0.count);
  }
  return n;
}

// Given a decoded primitive's geometry (LOD0 in geometry.index) and the coarse
// index typed-array (from the accessor referenced by extras.coarseIndexAccessor),
// produce ONE concatenated element buffer [LOD0 | coarse] and attach it as the
// geometry index, returning {clusterSet, lod0Count}. Tolerates absent extras
// (returns null -> caller renders the geometry as a plain full-res mesh).
export function attachClusterLod(geometry, extras, coarseIndexArray) {
  const clusterSet = parseClusterLod(extras);
  if (!clusterSet) return null;

  const lod0 = geometry.index ? geometry.index.array : null;
  if (!lod0) return null;
  const lod0Count = lod0.length;
  const coarse = coarseIndexArray || new Uint32Array(0);

  // One element buffer big enough for both; promote to Uint32 if needed.
  const maxVid = geometry.attributes.position.count - 1;
  const Ctor = maxVid > 65535 ? Uint32Array : Uint16Array;
  const combined = new Ctor(lod0Count + coarse.length);
  combined.set(lod0, 0);
  combined.set(coarse, lod0Count);
  geometry.setIndex(new THREE.BufferAttribute(combined, 1));

  return { clusterSet, lod0Count };
}
