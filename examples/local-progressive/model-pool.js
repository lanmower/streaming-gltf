// ModelPool — managed LOD streaming for skinned + morph-target meshes.
//
// Responsibilities:
//  - Load a baked progressive asset bundle (root GLB + sibling LOD files +
//    the extensions.EP_progressive_lod payload, legacy extras.LOCAL_progressive
//    still read as a fallback) at most ONCE per source URL.
//  - Share BufferGeometry and Texture instances across every Entity that
//    spawns from the same asset, so 1000 LANMOWERs reuse one geometry per LOD
//    rather than allocating 1000 copies.
//  - Spawn lightweight Entity handles (SkinnedMesh / Mesh wrappers) wired to
//    the right shared resources for their current LOD.
//  - Run one per-frame update that walks every live Entity, picks an LOD by
//    screen-space density + global ceiling, evicts/fetches as needed.
//  - Emit events ('ready', 'lod-changed', 'evicted', 'budget-pressure',
//    'fps') so application code can react without polling.
//
// Phase A scope: load-once, instance-share, event-driven Entity API.
// Phases B & C bolt onto this without changing the public surface.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
// Pure-JS Draco decoder (mrdoob draco.js), vendored at ./draco-loader.js — a
// drop-in for three's DRACOLoader that needs no .wasm fetch and no gstatic CDN.
import { DRACOLoader } from './draco-loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
// KTX2/Basis transcoder -> decodes the single per-texture KTX2 into a GPU-
// COMPRESSED texture (BCn/ASTC/ETC2 per device); the basis wasm is vendored at
// ./basis/ (served locally, no CDN).
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { GlobalMaterialPool } from './material-pool.js';
// Cluster-LOD (EP_cluster_lod): UV-aware spatial meshlet clusters with per-cluster
// hierarchical LODs packed into one unified buffer, drawn per-cluster via
// WEBGL_multi_draw. When an asset carries the extra this is THE LOD mechanism for
// it (supersedes the discrete _pickMeshLod sibling-LOD ladder); legacy assets
// without it keep the discrete path.
import { ClusterLodMesh, attachClusterLod } from './cluster-lod-mesh.js';
import { CLUSTER_LOD_EXTRA_KEY } from './meshlet-codec.js';
import { DeferredLoadQueue } from './deferred-load-queue.js';
import { LodUnloadManager } from './lod-unload-manager.js';
import { CachedFrustumPlanes } from './frustum-cache.js';
import { BatchedFarTier } from './batched-far-tier.js';
import { OctahedralImpostorTier } from './octahedral-impostor-tier.js';
import { OctahedralImpostorEzTier } from './octahedral-impostor-ez-tier.js';
import { OcclusionQueryTier } from './occlusion-query-tier.js';
// WebGpuHizTier is dynamic-imported (see _getOcclusionTier) rather than
// statically imported here: it depends on 'three/tsl', a subpath most
// existing demo pages' importmaps don't map (they only import 'three' for
// the plain WebGLRenderer path) -- a static import here broke every
// WebGL2-only demo page with "Failed to resolve module specifier three/tsl"
// even though those pages never construct a WebGPURenderer and never need
// this tier at all. Deferring the import until useOcclusionQuery is actually
// enabled AND the renderer looks like a WebGPURenderer keeps the WebGL2 path
// fully independent of the WebGPU tier's dependencies.
// Phase 3 Quick-Wins optimizations
import { VertexCompressionOptimizer } from './vertex-compression.js';

// draco.js decodes in pure JS — no decoder path / wasm to configure.
const _sharedDracoLoader = new DRACOLoader();

// Shared KTX2Loader, configured once with the renderer (detectSupport needs the
// GL context to pick the device's compressed format). Created lazily on first
// pool construction; _makeLoader attaches it so KHR_texture_basisu textures
// transcode to GPU-compressed on load.
let _sharedKtx2Loader = null;
function _ensureKtx2Loader(renderer) {
  if (_sharedKtx2Loader || !renderer) return _sharedKtx2Loader;
  _sharedKtx2Loader = new KTX2Loader()
    .setTranscoderPath(new URL('./basis/', import.meta.url).href)
    .detectSupport(renderer);
  return _sharedKtx2Loader;
}

// --- scratch objects (per-frame; never alloc in hot path) -----------------
const _tmpV3 = new THREE.Vector3();
const _tmpV3b = new THREE.Vector3();
const _tmpSphere = new THREE.Sphere();
const _zeroMatrix = new THREE.Matrix4().set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const _identityMatrix = new THREE.Matrix4(); // default = identity

// Tight, correct bounding-radius scale multiplier for a (possibly non-uniform)
// scale vector: max(|sx|,|sy|,|sz|). A uniform scale s previously computed via
// scale.length()/Math.SQRT2 yielded ~1.22s (22% over-inflated bound, wasted
// cull efficiency); a strongly non-uniform scale like (2,0.1,0.1) yielded
// ~1.42, UNDER-estimating the true extent (2) -- an actual false-cull bug.
function _maxAbsScale(scale) {
  return Math.max(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z));
}

// --- shared GLTFLoaders ---------------------------------------------------
// Two flavors: one with the VRM plugin (root loads), one without (sibling
// LOD loads — the siblings carry no VRM extension blob, and the plugin's
// MToon prep has side effects on attribute layout we don't want).
function _makeLoader(includeVrm) {
  const l = new GLTFLoader();
  l.setMeshoptDecoder(MeshoptDecoder);
  l.setDRACOLoader(_sharedDracoLoader);
  if (_sharedKtx2Loader) l.setKTX2Loader(_sharedKtx2Loader);
  if (includeVrm) l.register((parser) => new VRMLoaderPlugin(parser));
  return l;
}

// --- per-instance VRM parse gate ------------------------------------------
// @pixiv/three-vrm v3 has no skeleton-rebind clone, so to drive N independent
// instances of one VRM asset we re-parse the source bytes once per owning
// entity (each parse yields its own humanoid/springBone/expression managers
// bound to its own scene). Parsing is heavy, so bound concurrency to avoid a
// thundering N-player parse stall; excess parses queue and run as slots free.
const _MAX_VRM_PARSE_CONCURRENT = 4;
let _vrmParseActive = 0;
const _vrmParseQueue = [];
function _acquireVrmParseSlot() {
  if (_vrmParseActive < _MAX_VRM_PARSE_CONCURRENT) { _vrmParseActive++; return Promise.resolve(); }
  return new Promise((r) => _vrmParseQueue.push(r));
}
function _releaseVrmParseSlot() {
  _vrmParseActive--;
  const next = _vrmParseQueue.shift();
  if (next) { _vrmParseActive++; next(); }
}
// Parse an independent VRM from raw GLB bytes through a VRM-registered loader,
// returning { vrm, scene } where scene is THIS instance's own (un-shared) tree.
// removeUnnecessaryVertices/combineSkeletons mirror the single-owner prep so
// every driven instance matches the original's vertex/skeleton layout.
async function _parseOwnVrm(rootBytes) {
  await _acquireVrmParseSlot();
  try {
    const loader = _makeLoader(true);
    const gltf = await new Promise((resolve, reject) => {
      loader.parse(rootBytes.buffer.slice(rootBytes.byteOffset, rootBytes.byteOffset + rootBytes.byteLength), '', resolve, reject);
    });
    const vrm = gltf.userData?.vrm || null;
    if (vrm) {
      try { VRMUtils.removeUnnecessaryVertices(vrm.scene); } catch (e) { /* best-effort */ }
      try { VRMUtils.combineSkeletons(vrm.scene); } catch (e) { /* best-effort */ }
    }
    return { vrm, scene: vrm ? vrm.scene : gltf.scene };
  } finally {
    _releaseVrmParseSlot();
  }
}

// --- tiny EventEmitter ----------------------------------------------------
class Emitter {
  constructor() { this._listeners = new Map(); }
  on(ev, fn) {
    let s = this._listeners.get(ev);
    if (!s) { s = new Set(); this._listeners.set(ev, s); }
    s.add(fn);
    return () => s.delete(fn);
  }
  emit(ev, payload) {
    const s = this._listeners.get(ev);
    if (!s) return;
    for (const fn of s) {
      try { fn(payload); } catch (e) { console.error(`[ModelPool] listener for ${ev} threw`, e); }
    }
  }
}

// --- InstancedPool: one shared InstancedMesh per (asset, lod) -------------
// For the unskinned LOD tier we don't need per-entity skeletons or per-entity
// SkinnedMesh shells; the mesh is in bind pose and only its TRANSFORM differs
// across entities. Wrapping them all in one InstancedMesh collapses N draw
// calls into 1, which is the only realistic path to 1000+ entities on
// commodity hardware.
class InstancedSlot {
  constructor(pool, asset, meshDescIdx, lodIdx, geo, material) {
    this.pool = pool;
    this.asset = asset;
    this.meshDescIdx = meshDescIdx;
    this.lodIdx = lodIdx;
    this.geometry = geo;
    this.material = material;
    this.capacity = 32; // grow as needed
    // Per-frame uniform — ModelPool.update writes the camera's
    // projection*view matrix into here so the vertex shader can do GPU
    // frustum culling without a CPU sphere test per entity.
    this._uniforms = { projViewMatrix: { value: new THREE.Matrix4() } };
    // Initialize frustum plane cache (shared across all slots in this pool)
    if (!pool._frustumCache) pool._frustumCache = new CachedFrustumPlanes();
    this._uniforms.frustumPlanes = { value: pool._frustumCache.getPlaneUniforms() };
    // GPU-driven per-instance transform: a float DataTexture holds each
    // instance's model matrix as 4 RGBA texels (one mat4 column per texel),
    // instance i -> texels [i*4 .. i*4+3]. The vertex shader rebuilds the
    // matrix from gl_InstanceID, so JS never re-uploads a full instance buffer
    // per frame; a single model move is one 4-texel write + a dirty flag.
    // Each slot needs its OWN instanceTex uniform, so when the GPU path is on
    // the slot must use a PER-SLOT material (the shared global FAR material
    // could only bind one slot's texture). Each slot is already its own
    // InstancedMesh = its own draw, so cloning the material adds no draw call.
    this._gpuInstanceTex = pool._enableGpuInstanceTex !== false;
    if (this._gpuInstanceTex) {
      material = material.clone();
      this._initInstanceTexture(this.capacity);
    }
    _patchInstancedSlotMaterial(material, this._uniforms);
    this.material = material;
    this.mesh = new THREE.InstancedMesh(geo, material, this.capacity);
    this.mesh.frustumCulled = false; // GPU vertex-shader handles culling
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Per-instance world-space bounding-sphere RADIUS only (r). The center is
    // always exactly the instance's own matrix translation column (verified:
    // every write site below passes me[12..14] straight from the instance's
    // world matrix, never an offset center) so the vertex shader derives it
    // from instanceMatrix[3].xyz / readInstanceMatrix(...)[3].xyz instead of
    // carrying a redundant xyz here — 4x smaller attribute + upload. Set on
    // slot acquire / update; the vertex shader reads this and collapses
    // out-of-frustum instances to NaN.
    this._boundArray = new Float32Array(this.capacity);
    this._boundAttr = new THREE.InstancedBufferAttribute(this._boundArray, 1);
    this._boundAttr.setUsage(THREE.DynamicDrawUsage);
    this.mesh.geometry.setAttribute('instanceBoundSphere', this._boundAttr);
    // Dirty-range tracking for the bound-sphere attribute, mirroring the instance
    // transform texture's addUpdateRange treatment: a single moving instance's
    // sphere write otherwise re-uploads the WHOLE capacity*4 buffer every frame
    // (see flushInstanceTexture's comment for the identical bug class).
    this._boundDirtyRuns = [];
    // Zero out all instance matrices initially so unused slots draw nothing
    // visible (zero matrix collapses to origin point).
    const zero = new THREE.Matrix4().set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    for (let i = 0; i < this.capacity; i++) this.mesh.setMatrixAt(i, zero);
    this.mesh.count = 0;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.slots = new Map(); // entity -> slot index
    this.freeSlots = []; // recycled indices
    this.nextSlot = 0;
    this._dirtySlots = new Set(); // tracks which slot indices need GPU upload
  }

  acquireSlot(entity) {
    let idx;
    if (this.freeSlots.length) idx = this.freeSlots.pop();
    else {
      if (this.nextSlot >= this.capacity) this._grow(this.capacity * 2);
      idx = this.nextSlot++;
    }
    this.slots.set(entity, idx);
    if (idx + 1 > this.mesh.count) this.mesh.count = idx + 1;
    return idx;
  }
  releaseSlot(entity) {
    const idx = this.slots.get(entity);
    if (idx == null) return;
    this.slots.delete(entity);
    this.freeSlots.push(idx);
    // Zero its matrix so it stops drawing (collapses to origin / degenerate).
    const zero = _zeroMatrix;
    if (this._gpuInstanceTex) {
      this.setInstanceTransform(idx, zero);
    } else {
      this.mesh.setMatrixAt(idx, zero);
      this._dirtySlots.add(idx);
    }
    // Zero the bound-sphere radius so the shader treats this slot as
    // "no bound info" → also drawn at origin (zero matrix). Belt+braces.
    this._boundArray[idx] = 0;
    this._markBoundDirty(idx);
  }
  setMatrixForSlot(idx, matrix) {
    if (this._gpuInstanceTex) {
      // GPU path: write the matrix into the instance data texture. The shader
      // reads it by gl_InstanceID; we do not touch the instanceMatrix attribute.
      this.setInstanceTransform(idx, matrix);
      return;
    }
    this.mesh.setMatrixAt(idx, matrix);
    this._dirtySlots.add(idx);
  }
  // Optimization 2: Deferred matrix buffer uploads
  // Only mark needsUpdate if dirty slots exceed threshold (5-10% of capacity)
  // This reduces GPU buffer sync stalls by batching updates across multiple frames
  flushMatrixUpdates() {
    this._flushBoundAttr();
    if (this._gpuInstanceTex) { this.flushInstanceTexture(); return; }
    if (this._dirtySlots.size > 0) {
      // ALWAYS flush when there are dirty slots. The old 5%-of-capacity gate
      // skipped the GPU upload for small dirty counts but cleared _dirtySlots
      // anyway, so a released/moved instance's matrix sat un-uploaded in the CPU
      // buffer for frames — producing ghost models that pop in/out (most visible
      // when the zoom-cycle camera transitions a few entities' LOD at a time).
      this.mesh.instanceMatrix.needsUpdate = true;
      this._dirtySlots.clear();
    }
  }
  // Center is intentionally NOT stored here — it is always the instance's own
  // matrix translation, which the vertex shader already has via instanceMatrix
  // / the instance transform texture. Only the radius is CPU-tracked.
  setBoundSphereForSlot(idx, r) {
    this._boundArray[idx] = r;
    this._markBoundDirty(idx);
  }
  // Insert instance idx's touched component range into a merged disjoint-run
  // list (identical shape to _markInstanceTexDirty) so N scattered per-frame
  // movers upload O(N) components instead of O(capacity) components.
  _markBoundDirty(idx) {
    const loComp = idx, hiComp = idx;
    const runs = this._boundDirtyRuns;
    let i = 0;
    while (i < runs.length && runs[i][1] < loComp - 1) i++;
    let mergedLo = loComp, mergedHi = hiComp;
    let j = i;
    while (j < runs.length && runs[j][0] <= hiComp + 1) {
      if (runs[j][0] < mergedLo) mergedLo = runs[j][0];
      if (runs[j][1] > mergedHi) mergedHi = runs[j][1];
      j++;
    }
    // PERF: the common case (a moving instance touching/extending an already-
    // dirty run) has j > i, i.e. at least one existing [lo,hi] tuple is being
    // replaced — reuse that tuple's array in place instead of splice()'ing in
    // a freshly allocated 2-element array every call. Only allocate when this
    // is a genuinely new, disjoint run (j === i, nothing to reuse).
    if (j > i) {
      const tuple = runs[i];
      tuple[0] = mergedLo; tuple[1] = mergedHi;
      if (j - i > 1) runs.splice(i + 1, j - i - 1);
    } else {
      runs.splice(i, 0, [mergedLo, mergedHi]);
    }
  }
  // Upload only the touched component runs via addUpdateRange instead of a
  // full-buffer needsUpdate re-upload every frame any instance's bound sphere
  // changes (same fix class/rationale as flushInstanceTexture above).
  _flushBoundAttr() {
    const runs = this._boundDirtyRuns;
    if (runs.length > 0) {
      if (typeof this._boundAttr.addUpdateRange === 'function') {
        this._boundAttr.clearUpdateRanges();
        for (const [lo, hi] of runs) this._boundAttr.addUpdateRange(lo, hi - lo + 1);
      }
      this._boundAttr.needsUpdate = true;
      runs.length = 0;
    }
  }
  // --- GPU instance transform texture --------------------------------------
  // Texture layout: width = capacity*4 texels (4 per instance = a mat4's four
  // columns), height = 1. RGBA32F. instance i occupies texels [i*4 .. i*4+3].
  _initInstanceTexture(capacity) {
    const texelsPerInstance = 4;
    this._instTexWidth = capacity * texelsPerInstance;
    this._instTexData = new Float32Array(this._instTexWidth * 4);
    const tex = new THREE.DataTexture(this._instTexData, this._instTexWidth, 1, THREE.RGBAFormat, THREE.FloatType);
    // NearestFilter: exact texel reads + avoids OES_texture_float_linear
    // requirement (linear-filtering a float texture raises GL_INVALID_OPERATION).
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    this._instTex = tex;
    // Reuse existing uniform objects on regrow so the material's captured
    // references (from onBeforeCompile) stay valid — only swap their .value.
    if (this._uniforms.instanceTex) {
      this._uniforms.instanceTex.value = tex;
      this._uniforms.instanceTexWidth.value = this._instTexWidth;
    } else {
      this._uniforms.instanceTex = { value: tex };
      this._uniforms.instanceTexWidth = { value: this._instTexWidth };
    }
    // Dirty-range tracking for partial uploads: a SORTED SET of disjoint [loCol,hiCol]
    // texel-column runs touched since the last flush (see setInstanceTransform/flushInstanceTexture
    // for why this replaced a single min..max span).
    this._instTexDirtyRuns = [];
  }
  // Write one instance's mat4 into its 4 texels. Marks only that instance's
  // column range dirty — a single model move costs one 4-texel write here.
  setInstanceTransform(idx, matrix) {
    const e = matrix.elements; // column-major 16 floats
    const base = idx * 4 * 4; // 4 texels * 4 channels
    // column c -> texel (idx*4 + c) -> data[base + c*4 .. +3]
    for (let c = 0; c < 4; c++) {
      const o = base + c * 4;
      const m = c * 4;
      this._instTexData[o] = e[m];
      this._instTexData[o + 1] = e[m + 1];
      this._instTexData[o + 2] = e[m + 2];
      this._instTexData[o + 3] = e[m + 3];
    }
    this._markInstanceTexDirty(idx * 4, idx * 4 + 3);
  }
  // Insert [loCol,hiCol] into the sorted disjoint-run list, merging with any overlapping/adjacent
  // run so the list stays O(distinct touched regions) rather than growing one entry per instance.
  _markInstanceTexDirty(loCol, hiCol) {
    const runs = this._instTexDirtyRuns;
    let i = 0;
    while (i < runs.length && runs[i][1] < loCol - 1) i++;
    let mergedLo = loCol, mergedHi = hiCol;
    let j = i;
    while (j < runs.length && runs[j][0] <= hiCol + 1) {
      if (runs[j][0] < mergedLo) mergedLo = runs[j][0];
      if (runs[j][1] > mergedHi) mergedHi = runs[j][1];
      j++;
    }
    // PERF: same in-place-reuse fix as _markBoundDirty above — avoid
    // allocating a fresh 2-element tuple on every instance write when an
    // existing run can be mutated in place instead.
    if (j > i) {
      const tuple = runs[i];
      tuple[0] = mergedLo; tuple[1] = mergedHi;
      if (j - i > 1) runs.splice(i + 1, j - i - 1);
    } else {
      runs.splice(i, 0, [mergedLo, mergedHi]);
    }
  }
  // Upload only the touched texel columns via THREE's addUpdateRange (three>=0.159) instead of a
  // full-width needsUpdate re-upload every frame something moved. PERF (2026-07-02, spoint consumer
  // 144fps investigation): a live stack-trace CDP profile traced texSubImage2D — a top-3 live-frame
  // cost — to three's uploadTexture->setTexture2D re-uploading this ENTIRE _instTexWidth-wide row
  // (capacity*4 texels) on every frame ANY tracked entity moved, even when only one instance's 4-texel
  // range actually changed. addUpdateRange narrows the GPU upload to the byte-exact dirty span
  // (start/count are in COMPONENTS: RGBAFormat = 4 components/texel, so texel range [lo,hi] ->
  // component range [lo*4, (hi-lo+1)*4]).
  //
  // FOLLOW-UP FIX (2026-07-02i): the original version tracked a single min..max SPAN across all
  // dirty instances in a frame. When co-moving entities land at scattered pool slot indices (the
  // normal case — slot assignment is allocation-order, not spatial/temporal locality), that span
  // silently widens to cover nearly the whole texture, degrading back to a near-full-width upload
  // with no signal that the "partial" path stopped helping. Now tracks a merged list of disjoint
  // dirty runs and issues one addUpdateRange per run, so N scattered movers upload O(N) texels
  // instead of O(capacity) texels regardless of how spread out their slot indices are.
  flushInstanceTexture() {
    const runs = this._instTexDirtyRuns;
    if (runs.length > 0) {
      if (typeof this._instTex.addUpdateRange === 'function') {
        this._instTex.clearUpdateRanges();
        for (const [lo, hi] of runs) this._instTex.addUpdateRange(lo * 4, (hi - lo + 1) * 4);
      }
      this._instTex.needsUpdate = true;
      runs.length = 0;
    }
  }
  _grow(newCap) {
    const old = this.mesh;
    const next = new THREE.InstancedMesh(this.geometry, this.material, newCap);
    next.frustumCulled = false;
    next.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this._gpuInstanceTex) {
      // Grow the instance data texture, preserving existing instance matrices.
      const oldData = this._instTexData;
      this._initInstanceTexture(newCap);
      this._instTexData.set(oldData); // copy old texels into the front of the new buffer
      this._instTex.needsUpdate = true;
      // Re-point the shader uniform at the new texture (same uniform object the
      // material's onBeforeCompile captured, so just swap its .value).
      this._uniforms.instanceTex.value = this._instTex;
      this._uniforms.instanceTexWidth.value = this._instTexWidth;
    } else {
      const m = new THREE.Matrix4();
      for (let i = 0; i < this.nextSlot; i++) {
        old.getMatrixAt(i, m);
        next.setMatrixAt(i, m);
      }
      next.instanceMatrix.needsUpdate = true;
    }
    next.count = old.count;
    // Grow + carry the per-instance bound-sphere radius attribute.
    const newBounds = new Float32Array(newCap);
    newBounds.set(this._boundArray);
    this._boundArray = newBounds;
    this._boundAttr = new THREE.InstancedBufferAttribute(newBounds, 1);
    this._boundAttr.setUsage(THREE.DynamicDrawUsage);
    next.geometry.setAttribute('instanceBoundSphere', this._boundAttr);
    this._boundDirtyRuns = []; // fresh attribute object, no pending partial-range upload to carry
    const parent = old.parent;
    if (parent) {
      parent.remove(old);
      parent.add(next);
    }
    old.dispose();
    this.mesh = next;
    this.capacity = newCap;
    this._dirtySlots = new Set();
  }
}

// Patch a material so its vertex shader receives a per-instance bound-sphere
// attribute and a per-frame projViewMatrix uniform, then collapses any
// instance outside the camera frustum to a NaN clip-space position so the GPU
// early-rejects it. Frustum planes are pre-normalized on the CPU, so the cull
// is a branch-free dot product per plane with no per-vertex sqrt/divide.
// Wraps any existing onBeforeCompile so the vertex-color gamma patch on the
// fragment side still runs.
function _patchInstancedSlotMaterial(material, uniforms) {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (shader) => {
    if (prev) prev(shader);
    shader.uniforms.projViewMatrix = uniforms.projViewMatrix;
    // 6 pre-computed, unit-normalized frustum planes (normal.xyz + constant.w),
    // updated once per frame on the CPU. The vertex shader uses them directly.
    shader.uniforms.frustumPlanes = uniforms.frustumPlanes;
    // GPU instance transform texture (per-instance mat4 as 4 RGBA texels).
    if (uniforms.instanceTex) {
      shader.uniforms.instanceTex = uniforms.instanceTex;
      shader.uniforms.instanceTexWidth = uniforms.instanceTexWidth;
      shader.defines = shader.defines || {};
      shader.defines.USE_GPU_INSTANCE_TEX = '';
    }
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
attribute float instanceBoundSphere;
uniform mat4 projViewMatrix;
uniform vec4 frustumPlanes[6];
#ifdef USE_GPU_INSTANCE_TEX
uniform sampler2D instanceTex;
uniform float instanceTexWidth;
mat4 readInstanceMatrix(int id) {
  // 4 texels per instance; fetch by pixel center. height = 1.
  float base = float(id) * 4.0;
  vec4 c0 = texture2D(instanceTex, vec2((base + 0.5) / instanceTexWidth, 0.5));
  vec4 c1 = texture2D(instanceTex, vec2((base + 1.5) / instanceTexWidth, 0.5));
  vec4 c2 = texture2D(instanceTex, vec2((base + 2.5) / instanceTexWidth, 0.5));
  vec4 c3 = texture2D(instanceTex, vec2((base + 3.5) / instanceTexWidth, 0.5));
  return mat4(c0, c1, c2, c3);
}
#endif`
      )
      .replace(
        '#include <project_vertex>',
        `#ifdef USE_GPU_INSTANCE_TEX
  // GPU-driven transform: rebuild this instance's model matrix from the
  // instance data texture (by gl_InstanceID) instead of the instanceMatrix
  // attribute. mvPosition is declared at outer scope (exactly like the stock
  // <project_vertex> chunk) so downstream chunks that read it still compile.
  mat4 instMat = readInstanceMatrix(gl_InstanceID);
  vec4 mvPosition = modelViewMatrix * instMat * vec4(transformed, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  vec3 instCenter = instMat[3].xyz;
#else
  #include <project_vertex>
  vec3 instCenter = instanceMatrix[3].xyz;
#endif
{
  // GPU per-instance frustum cull.
  // frustumPlanes are pre-normalized CPU-side (THREE.Frustum emits unit
  // normals), so the plane equation reduces to dot(n, c) + w >= -r with no
  // per-vertex sqrt/divide. (Removed the old length()/division — it was
  // normalizing an already-unit vector. Also removed the dead lodLutTexture
  // fetch + vLodIndex varying: LOD selection happens CPU-side, the varying
  // was written but never read by any fragment shader.)
  // Center is NOT a separate attribute: it is always exactly the instance's
  // own model-matrix translation column, read directly from whichever
  // transform path is active above (verified CPU-side: every JS write site
  // passes the instance's own worldMat translation, never an offset center).
  if (instanceBoundSphere > 0.0) {
    vec3 c = instCenter;
    float r = instanceBoundSphere;
    bool outside = false;
    for (int i = 0; i < 6; i++) {
      vec4 p = frustumPlanes[i];
      if (dot(p.xyz, c) + p.w < -r) { outside = true; break; }
    }
    if (outside) {
      gl_Position = vec4(0.0/0.0, 0.0/0.0, 0.0/0.0, 0.0/0.0) * 0.0;
      return;
    }
  }
}`
      );
  };
  material.needsUpdate = true;
}

// --- Asset: shared resources for one source URL ---------------------------
// Loaded once, referenced by N Entity instances.
class Asset {
  constructor(pool, url) {
    this.pool = pool;
    this.url = url;
    this.state = 'pending'; // 'pending' | 'loading' | 'ready' | 'error'
    this.error = null;
    // baseDir is the directory of the root model.progressive.glb so we can
    // resolve sibling LOD relative paths against it.
    this.baseDir = url.endsWith('/') ? url : url.replace(/[^/]+$/, '');
    // Per-mesh LOD descriptors from the EP_progressive_lod payload, sorted
    // by quality ascending (idx 0 = lowest, idx N-1 = highest).
    this.meshLodDescs = []; // [{ meshIndex, primIndex, lods: [...] }]
    this.texLodDescs = []; // [{ textureIndex, name, lods: [...] }]
    // Cached shared geometries: key `${meshIndex}:${primIndex}:${lodIdx}` -> BufferGeometry
    this.geoCache = new Map();
    // Cached shared texture bitmaps: key `${textureIndex}:${lodIdx}` -> ImageBitmap
    this.texCache = new Map();
    // The original gltf payload from the root load — used to clone scenes
    // per-entity. Held as the parsed three.js Object3D plus parser.json.
    this.rootGltf = null;
    // Raw GLB bytes, retained for VRM assets only so each driven entity can
    // re-parse its OWN independent VRM runtime (three-vrm v3 has no
    // skeleton-rebind clone, so sharing one parsed VRM across instances would
    // freeze all-but-one in bind pose). Null for non-VRM assets.
    this.rootBytes = null;
    this.hasVRM = false;
    // Track bytes-loaded per LOD for the budget system in Phase C.
    this.byteWeights = new Map(); // key -> bytes
    // Loaders need to know whether this asset is VRM-bearing (root) or a
    // plain sibling LOD (no VRM).
    this._rootLoader = _makeLoader(true);
    this._lodLoader = _makeLoader(false);
    // Promise that resolves when the root is parsed.
    this.ready = this._load();
  }

  async _fetchBytes(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    // A 200 response isn't necessarily a valid GLB (HTML error page served with
    // 200, truncated download, wrong content-type from a misconfigured host) --
    // catch that here with a clear message naming the URL, instead of letting
    // GLTFLoader.parse() throw a low-level/cryptic binary-parsing error later
    // that gives no hint the real problem was the server response, not the model.
    if (buf.byteLength < 4 || (buf[0] !== 0x67 || buf[1] !== 0x6c || buf[2] !== 0x54 || buf[3] !== 0x46)) {
      throw new Error(`fetch ${url}: response is not a valid GLB (bad magic; got ${buf.byteLength} byte(s), possibly an HTML error page or truncated download)`);
    }
    this.pool._trackBytes(this.url, url, buf.byteLength);
    return buf;
  }

  async _load() {
    this.state = 'loading';
    try {
      const rootBytes = await this._fetchBytes(this.url);
      const gltf = await new Promise((resolve, reject) => {
        this._rootLoader.parse(rootBytes.buffer, '', resolve, reject);
      });
      this.rootGltf = gltf;
      this.hasVRM = !!gltf.userData?.vrm;
      // Retain the raw GLB bytes ONLY for VRM assets: each driven entity
      // re-parses its own independent VRM runtime from these (multi-driver).
      // Non-VRM assets clone the parsed scene and never need the bytes again,
      // so we drop them to avoid pinning the buffer in memory.
      this.rootBytes = this.hasVRM ? rootBytes : null;
      // Prefer the conformant extensions[EP_progressive_lod] payload; fall
      // back to the legacy extras.LOCAL_progressive blob so assets baked before
      // the extension rename still load.
      const json = gltf.parser.json;
      const ext = json?.extensions?.EP_progressive_lod ?? json?.extras?.LOCAL_progressive;
      if (ext) {
        const kindRank = { unskinned: 0, vertcolor: 1, textured: 2 };
        for (const m of ext.meshes) {
          const sorted = [...m.lods].sort((a, b) => {
            const ra = kindRank[a.kind || 'textured'] ?? 2;
            const rb = kindRank[b.kind || 'textured'] ?? 2;
            if (ra !== rb) return ra - rb;
            return (a.ratio || 0) - (b.ratio || 0);
          });
          this.meshLodDescs.push({ meshIndex: m.meshIndex, primIndex: m.primIndex, lods: sorted });
        }
        for (const t of ext.textures) {
          const sortedT = [...t.lods].sort((a, b) => a.width - b.width);
          this.texLodDescs.push({ textureIndex: t.textureIndex, name: t.name, lods: sortedT });
        }
      }

      // Cluster-LOD detection: scan the glTF json for primitives carrying
      // extras.EP_cluster_lod, indexed by mesh-traversal order so Entity._bootstrap
      // can match the cloned scene's meshes. When ANY prim has it the asset is in
      // cluster mode and bypasses the discrete-LOD machinery for those meshes.
      this.clusterLod = null; // meshTraversalIdx -> { extras, coarseAccessorIndex }
      try {
        let any = false;
        const map = new Map();
        let order = 0;
        for (const m of json.meshes || []) {
          for (const prim of m.primitives || []) {
            const ex = prim.extras && prim.extras[CLUSTER_LOD_EXTRA_KEY];
            if (ex) { map.set(order, { extras: prim.extras, coarseAccessorIndex: ex.coarseIndexAccessor }); any = true; }
            order++;
          }
        }
        if (any) this.clusterLod = map;
      } catch (_) { this.clusterLod = null; }

      // Cluster mode: prepare each clustered mesh's geometry ONCE on the asset
      // (attach the combined [LOD0|coarse] index + parse the ClusterSet). Entities
      // share this geometry; each spawns its own ClusterLodMesh (the per-cluster
      // cull/LOD/multi-draw self-drives off camera in onBeforeRender). This is THE
      // LOD path — the discrete meshLodDescs ladder is not built in cluster mode.
      this.clusterMeshes = null; // [{ geometry, material, clusterSet, lod0Count }]
      if (this.clusterLod) {
        this.clusterMeshes = [];
        const meshes = [];
        gltf.scene.traverse((c) => { if (c.isMesh) meshes.push(c); });
        let ord = 0;
        for (const m of meshes) {
          const info = this.clusterLod.get(ord++);
          if (!info) continue;
          let coarse = null;
          if (info.coarseAccessorIndex >= 0) {
            try { coarse = (await gltf.parser.getDependency('accessor', info.coarseAccessorIndex)).array; } catch (_) {}
          }
          const attached = attachClusterLod(m.geometry, info.extras, coarse);
          if (attached) this.clusterMeshes.push({ geometry: m.geometry, material: m.material, clusterSet: attached.clusterSet, lod0Count: attached.lod0Count });
        }
      }
      // Pre-cache the inline (lowest-textured) geometry per primitive AND the
      // smallest textures from the root — they're already in the parsed
      // gltf scene; no second fetch needed.
      let meshIdx = 0;
      // ALL LODs are kept in mesh-LOCAL space (sibling LODs via the decodeAABB
      // remap in _bakeQuantizeDecode; the receiving tm.mesh / instanced slot
      // applies the world transform at render). The inline (LOD0) geometry must
      // follow the SAME convention — DO NOT bake matrixWorld into it, or assets
      // whose mesh node has a non-identity transform end up double/differently
      // transformed and visibly FLIP orientation when switching to a sibling LOD.
      gltf.scene.traverse((c) => {
        if (c.isMesh) {
          const desc = this.meshLodDescs[meshIdx];
          if (desc) {
            const inlineLodIdx = desc.lods.findIndex((l) => l.inline);
            if (inlineLodIdx >= 0) {
              this.geoCache.set(`${desc.meshIndex}:${desc.primIndex}:${inlineLodIdx}`, c.geometry);
            }
          }
          meshIdx++;
        }
      });
      // Cache the inline-sized texture bitmaps from the parsed scene.
      gltf.scene.traverse((c) => {
        if (!c.isMesh || !c.material) return;
        const slots = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap'];
        for (const s of slots) {
          const tex = c.material[s];
          if (!tex || !tex.image) continue;
          // Find which descriptor this texture belongs to (name match).
          const desc = this.texLodDescs.find((d) => d.name === tex.name);
          if (desc) {
            const inlineIdx = desc.lods.findIndex((l) => l.inline);
            if (inlineIdx >= 0) {
              this.texCache.set(`${desc.textureIndex}:${inlineIdx}`, tex.image);
            }
          }
        }
      });
      this.state = 'ready';
    } catch (e) {
      this.state = 'error';
      this.error = e;
      throw e;
    }
  }

  // Fetch a mesh LOD's shared geometry. Returns a Promise<BufferGeometry>.
  // Triggers the pool's per-asset request queue.
  async ensureMeshLod(meshDescIdx, lodIdx) {
    const desc = this.meshLodDescs[meshDescIdx];
    if (!desc) return null;
    const target = desc.lods[lodIdx];
    if (!target) return null;
    const key = `${desc.meshIndex}:${desc.primIndex}:${lodIdx}`;
    const cached = this.geoCache.get(key);
    if (cached) return cached;
    if (target.inline) return null; // should already be cached from root load
    // De-dupe in-flight requests through the pool's load queue (Phase C).
    return this.pool._enqueue(`${this.url}#${key}`, async () => {
      const stillCached = this.geoCache.get(key);
      if (stillCached) return stillCached;
      const fullUrl = this.baseDir + target.path;
      // Worker path: fetch + parse + bake-decode all happen off-thread; we
      // get back a payload of transferable typed arrays and a small bbox
      // record. Main thread only allocates a BufferGeometry shell wiring
      // those arrays as attributes — no per-vertex JS loops.
      if (this.pool._workers.length) {
        try {
          // Far/unskinned LOD: ask the worker to sloppy-decimate to ~400 tris
          // at load (the shipped far LODs are ~6500 tris each = the dominant
          // triangle cost; no re-bake of the 953 assets needed).
          const sloppyCap = (target.kind === 'unskinned') ? (this.pool._farTriCap ?? 400) : 0;
          const payload = await this.pool._workerFetchLod(fullUrl, target.decodeAABB, sloppyCap);
          this.pool._trackBytes(this.url, fullUrl, payload.bytes);
          let geo = ModelPool._buildGeometryFromPayload(payload);
          // Phase 3 QW1: Apply vertex compression (vec4 → vec3)
          geo = this.pool._compressGeometryAttributes(geo);
          // Phase 3 QW5: Apply attribute deinterleaving
          geo = this.pool._deinterleaveGeometryAttributes(geo);
          // Guard: ensure the far/unskinned LOD is capped even if the worker's
          // decimation didn't apply (stale worker, thrown pass, etc.). Idempotent
          // (returns early when already <= cap). This is the single choke point.
          if (target.kind === 'unskinned') _clusterDecimate(geo, this.pool._farTriCap ?? 400);
          this.geoCache.set(key, geo);
          this.byteWeights.set(key, payload.bytes);
          return geo;
        } catch (e) {
          // Fall through to main-thread path on worker failure.
          console.warn('[asset] worker decode failed, fallback main thread', e);
        }
      }
      // Main-thread fallback (or workerCount: 0).
      try {
        const bytes = await this._fetchBytes(fullUrl);
        const gltf = await new Promise((resolve, reject) => {
          this._lodLoader.parse(bytes.buffer, '', resolve, reject);
        });
        let srcMesh = null;
        gltf.scene.updateMatrixWorld(true);
        gltf.scene.traverse((c) => { if (c.isMesh && !srcMesh) srcMesh = c; });
        let geo = srcMesh?.geometry;
        if (geo) {
          _bakeQuantizeDecode(geo, srcMesh.matrixWorld, target.decodeAABB);
          // Phase 3 QW1: Apply vertex compression (vec4 → vec3)
          geo = this.pool._compressGeometryAttributes(geo);
          // Phase 3 QW5: Apply attribute deinterleaving
          geo = this.pool._deinterleaveGeometryAttributes(geo);
          // Cap the far/unskinned LOD AFTER compress/deinterleave so nothing
          // downstream can restore the full-res index (single choke point,
          // matches the worker path's guard).
          if (target.kind === 'unskinned') _clusterDecimate(geo, this.pool._farTriCap ?? 400);
          this.geoCache.set(key, geo);
          this.byteWeights.set(key, bytes.byteLength);
        }
        return geo;
      } catch (e) {
        console.warn(`[asset] LOD mesh ${key} failed to load (${e.message}), using previous LOD`);
        // Graceful fallback: return cached LOD from lower detail level
        if (lodIdx > 0) {
          return await this.ensureMeshLod(meshDescIdx, lodIdx - 1);
        }
        return null; // no fallback available at LOD 0
      }
    });
  }

  async ensureTexLod(texDescIdx, lodIdx) {
    const desc = this.texLodDescs[texDescIdx];
    if (!desc) return null;
    const target = desc.lods[lodIdx];
    if (!target) return null;
    const key = `${desc.textureIndex}:${lodIdx}`;
    const cached = this.texCache.get(key);
    if (cached) return cached;
    if (target.inline) return null;
    return this.pool._enqueue(`${this.url}#tex:${key}`, async () => {
      const stillCached = this.texCache.get(key);
      if (stillCached) return stillCached;
      const bytes = await this._fetchBytes(this.baseDir + target.path);
      const blob = new Blob([bytes], { type: target.mime || 'image/webp' });
      const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
      this.texCache.set(key, bmp);
      this.byteWeights.set(`tex:${key}`, bytes.byteLength);
      return bmp;
    });
  }

  // Evict a LOD's cached resource (Phase C). Called by the pool.
  evictMeshLod(meshDescIdx, lodIdx) {
    const desc = this.meshLodDescs[meshDescIdx];
    if (!desc) return false;
    const key = `${desc.meshIndex}:${desc.primIndex}:${lodIdx}`;
    const target = desc.lods[lodIdx];
    if (target?.inline) return false; // never evict inline geometry
    const geo = this.geoCache.get(key);
    if (!geo) return false;
    geo.dispose();
    this.geoCache.delete(key);
    this.byteWeights.delete(key);
    return true;
  }
  evictTexLod(texDescIdx, lodIdx) {
    const desc = this.texLodDescs[texDescIdx];
    if (!desc) return false;
    const key = `${desc.textureIndex}:${lodIdx}`;
    const target = desc.lods[lodIdx];
    if (target?.inline) return false;
    const bmp = this.texCache.get(key);
    if (!bmp) return false;
    if (bmp.close) bmp.close();
    this.texCache.delete(key);
    this.byteWeights.delete(`tex:${key}`);
    return true;
  }

  // Full teardown of the shared root payload. Per-entity dispose() never
  // touches this (non-VRM entities clone rootGltf.scene; VRM entities parse
  // their own); call this only when the asset itself is evicted from the pool.
  // For VRM assets the rootGltf is the asset's OWN parse (not handed to any
  // entity), so deepDispose frees its managers; entity VRM scenes are freed in
  // Entity.dispose. Dropping rootBytes releases the retained GLB buffer.
  dispose() {
    const vrm = this.rootGltf?.userData?.vrm;
    if (vrm) {
      try { VRMUtils.deepDispose(this.rootGltf.scene); } catch (e) { /* best-effort */ }
    }
    for (const geo of this.geoCache.values()) geo?.dispose?.();
    for (const bmp of this.texCache.values()) bmp?.close?.();
    this.geoCache.clear();
    this.texCache.clear();
    this.byteWeights.clear();
    this.rootBytes = null;
    this.rootGltf = null;
    this.state = 'disposed';
  }
}

// Repack interleaved attributes into standalone Float32 and bake the source
// mesh's local matrix (which carries the dequantize transform for plain
// Meshes loaded via GLTFLoader+KHR_mesh_quantization) into vertex data.
// When the matrix is identity, fall back to scanning the actual post-
// dequantize range and remapping into the per-LOD decodeAABB captured at
// bake time. This is the same logic the inline demo used; lifted into a
// helper so the pool and any direct consumers can share it.
// Dependency-free far-LOD decimation by spatial-grid vertex clustering (main-
// thread copy of the worker's _clusterDecimate, for the fallback load path).
// Caps the far/unskinned LOD to ~triCap triangles — invisible on a distant dot.
function _clusterDecimate(geo, triCap) {
  const pos = geo.attributes.position;
  if (!pos) return;
  // Handle non-indexed geometry by synthesizing a sequential index (this was a
  // straggler cause — some far LODs arrive non-indexed and the old !ix guard
  // skipped them, leaving 100k+ tris).
  let ix = geo.index;
  if (!ix) { const seq = new Uint32Array(pos.count); for (let i = 0; i < pos.count; i++) seq[i] = i; ix = { array: seq, count: pos.count }; }
  if (ix.count / 3 <= triCap) return;
  const idx = ix.array, px = pos.array, pStride = pos.itemSize;
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const x = px[i * pStride], y = px[i * pStride + 1], z = px[i * pStride + 2];
    if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y; if (z < mnz) mnz = z; if (z > mxz) mxz = z;
  }
  const sx = (mxx - mnx) || 1, sy = (mxy - mny) || 1, sz = (mxz - mnz) || 1;
  const nrm = geo.attributes.normal, col = geo.attributes.color;
  // Try FINE -> COARSE (high res first). Higher res = more cells = MORE kept
  // verts/tris; lower res = coarser = fewer. We want the FINEST grid that still
  // lands at/under the cap, so we descend and accept the first that fits, with
  // res=2 (8 cells, ~12 tris max) as the guaranteed-tiny floor. (The earlier
  // coarse->fine ascent was backwards: it left dense meshes stuck above the cap
  // and then accepted the worst, fine result — that was the 56-straggler bug.)
  for (let res = 48; res >= 2; res = (res > 8 ? res >> 1 : res - 2)) {
    const cellOf = new Int32Array(pos.count);
    const cellMap = new Map();
    let kept = 0;
    for (let i = 0; i < pos.count; i++) {
      const gx = Math.min(res - 1, ((px[i * pStride] - mnx) / sx * res) | 0);
      const gy = Math.min(res - 1, ((px[i * pStride + 1] - mny) / sy * res) | 0);
      const gz = Math.min(res - 1, ((px[i * pStride + 2] - mnz) / sz * res) | 0);
      const key = (gx * res + gy) * res + gz;
      let rep = cellMap.get(key);
      if (rep === undefined) { rep = kept++; cellMap.set(key, rep); }
      cellOf[i] = rep;
    }
    const out = [];
    for (let t = 0; t < idx.length; t += 3) {
      const a = cellOf[idx[t]], b = cellOf[idx[t + 1]], c = cellOf[idx[t + 2]];
      if (a !== b && b !== c && a !== c) out.push(a, b, c);
    }
    const outTris = out.length / 3;
    if (outTris <= triCap || res <= 2) {
      if (outTris < 1) continue; // too coarse (all degenerate) — try next finer? no, we descend; guard below
      const srcOf = new Int32Array(kept).fill(-1);
      for (let i = 0; i < pos.count; i++) { const r = cellOf[i]; if (srcOf[r] === -1) srcOf[r] = i; }
      const newPos = new Float32Array(kept * 3);
      const ct = col ? col.itemSize : 0;
      const newNrm = nrm ? new Float32Array(kept * 3) : null;
      const newCol = col ? new Float32Array(kept * ct) : null;
      for (let r = 0; r < kept; r++) {
        const s = srcOf[r];
        newPos[r * 3] = pos.getX(s); newPos[r * 3 + 1] = pos.getY(s); newPos[r * 3 + 2] = pos.getZ(s);
        // Getters denormalize normalized source attrs (Int8 normals, Uint8
        // colors) to floats — a raw .array copy left 0..255 colors -> WHITE.
        if (newNrm) { newNrm[r * 3] = nrm.getX(s); newNrm[r * 3 + 1] = nrm.getY(s); newNrm[r * 3 + 2] = nrm.getZ(s); }
        if (newCol) {
          newCol[r * ct] = col.getX(s);
          if (ct >= 2) newCol[r * ct + 1] = col.getY(s);
          if (ct >= 3) newCol[r * ct + 2] = col.getZ(s);
          if (ct >= 4) newCol[r * ct + 3] = col.getW(s);
        }
      }
      geo.setAttribute('position', new THREE.BufferAttribute(newPos, 3, false));
      if (newNrm) geo.setAttribute('normal', new THREE.BufferAttribute(newNrm, 3, false));
      if (newCol) geo.setAttribute('color', new THREE.BufferAttribute(newCol, ct, false));
      geo.setIndex(new THREE.BufferAttribute(kept > 65535 ? new Uint32Array(out) : new Uint16Array(out), 1));
      geo.computeBoundingSphere(); geo.computeBoundingBox();
      return;
    }
  }
}

function _bakeQuantizeDecode(geo, matrix, decodeAABB) {
  // Prefer the AABB-remap path: it places vertices in mesh-LOCAL space, the
  // same coordinate convention the inline (baseline) LOD uses. Applying the
  // sibling LOD's matrixWorld here would double-transform when the receiving
  // tm.mesh later composes its own (non-identity) world matrix during render.
  // The matrix path is a fallback for legacy bakes without decodeAABB.
  const m = matrix;
  const isIdentity = !decodeAABB && (
    m.elements[0] === 1 && m.elements[5] === 1 && m.elements[10] === 1 &&
    m.elements[12] === 0 && m.elements[13] === 0 && m.elements[14] === 0 &&
    m.elements[1] === 0 && m.elements[2] === 0 && m.elements[4] === 0 &&
    m.elements[6] === 0 && m.elements[8] === 0 && m.elements[9] === 0
  );
  if (!decodeAABB && !isIdentity) {
    for (const semKey of ['position', 'normal', 'tangent']) {
      const a = geo.attributes[semKey];
      if (!a) continue;
      const out = new Float32Array(a.count * a.itemSize);
      for (let i = 0; i < a.count; i++) {
        if (a.itemSize >= 1) out[i * a.itemSize + 0] = a.getX(i);
        if (a.itemSize >= 2) out[i * a.itemSize + 1] = a.getY(i);
        if (a.itemSize >= 3) out[i * a.itemSize + 2] = a.getZ(i);
        if (a.itemSize >= 4) out[i * a.itemSize + 3] = a.getW(i);
      }
      geo.setAttribute(semKey, new THREE.BufferAttribute(out, a.itemSize, false));
    }
    geo.applyMatrix4(m);
  } else if (decodeAABB) {
    const { min, max } = decodeAABB;
    const pos = geo.attributes.position;
    if (pos) {
      let smnX = Infinity, smxX = -Infinity;
      let smnY = Infinity, smxY = -Infinity;
      let smnZ = Infinity, smxZ = -Infinity;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        if (x < smnX) smnX = x; if (x > smxX) smxX = x;
        if (y < smnY) smnY = y; if (y > smxY) smxY = y;
        if (z < smnZ) smnZ = z; if (z > smxZ) smxZ = z;
      }
      const r = (a, b) => (b - a < 1e-9 ? 1 : b - a);
      const sx = (max[0] - min[0]) / r(smnX, smxX);
      const sy = (max[1] - min[1]) / r(smnY, smxY);
      const sz = (max[2] - min[2]) / r(smnZ, smxZ);
      const out = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        out[i * 3 + 0] = (pos.getX(i) - smnX) * sx + min[0];
        out[i * 3 + 1] = (pos.getY(i) - smnY) * sy + min[1];
        out[i * 3 + 2] = (pos.getZ(i) - smnZ) * sz + min[2];
      }
      geo.setAttribute('position', new THREE.BufferAttribute(out, 3, false));
      for (const semKey of ['normal', 'tangent']) {
        const a = geo.attributes[semKey];
        if (!a) continue;
        const o = new Float32Array(a.count * a.itemSize);
        for (let i = 0; i < a.count; i++) {
          if (a.itemSize >= 1) o[i * a.itemSize + 0] = a.getX(i);
          if (a.itemSize >= 2) o[i * a.itemSize + 1] = a.getY(i);
          if (a.itemSize >= 3) o[i * a.itemSize + 2] = a.getZ(i);
          if (a.itemSize >= 4) o[i * a.itemSize + 3] = a.getW(i);
        }
        geo.setAttribute(semKey, new THREE.BufferAttribute(o, a.itemSize, false));
      }
    }
  }
  geo.computeBoundingSphere();
  geo.computeBoundingBox();
}

// --- Entity: one live instance --------------------------------------------
// Encapsulates the per-instance THREE.Object3D tree (root Object3D containing
// SkinnedMesh / Mesh + skeleton clone), per-mesh LOD state, and per-frame
// update logic.
class Entity extends Emitter {
  constructor(pool, asset, opts) {
    super();
    this.pool = pool;
    this.asset = asset;
    this.id = ++pool._nextEntityId;
    this.opts = opts || {};
    // Root container — application code can `.add()` it to a scene, set
    // position/rotation/scale on it, etc.
    this.root = new THREE.Object3D();
    this.root.name = `entity_${this.id}_${asset.url.split('/').pop()}`;
    if (opts.position) this.root.position.fromArray(opts.position);
    if (opts.rotation) this.root.quaternion.setFromEuler(new THREE.Euler().fromArray(opts.rotation));
    if (opts.scale) this.root.scale.setScalar(opts.scale);
    // Caller can pass `static: true` for entities that never move after
    // spawn — we then disable auto matrix updates after composing position
    // into matrix once. Subsequent frames skip the matrix recompute walk.
    // Critical: matrixAutoUpdate=false means three.js's render-time
    // updateMatrixWorld skips updateMatrix(), so we must call it manually
    // here while position/quaternion/scale are still being read into matrix.
    if (opts.static) {
      this.root.updateMatrix(); // compose position/quat/scale into matrix
      this.root.matrixAutoUpdate = false;
      this.root.matrixWorldNeedsUpdate = true; // force one world recompute
    }
    // Per-mesh tracking: each tracks current LOD, the live SkinnedMesh/Mesh,
    // and per-tex current LOD.
    this.trackedMeshes = []; // [{ meshDescIdx, currentLod, mesh, texState: [{currentLod}, ...], baseSkeleton, baseMaterial, sharedTextures }]
    // Animation state.
    this.animationMixer = null;
    this.animationClips = [];
    this.animationAction = null;
    // VRM runtime (shared across LODs of one entity).
    this.vrm = null;
    // Frustum culling cached state.
    this._lastInFrustum = true;
    this._cachedFrustumVisible = true; // Boolean cache: entity is in frustum (OPTIMIZATION)
    this._frustumCheckInterval = 0; // Frame counter: test frustum every 2-3 frames or when entity moves
    this._firstFrustumTest = true; // Force frustum test on first update to avoid "disappear" bug
    // Screen-space pixel size cache for tier allocation (updated only when entity moves)
    this._lastScreenPx = null;
    // Cached flag: whether all tracked meshes are in instanced slots.
    this._allInstanced = false;
    // Disposed flag — stop touching this entity after dispose().
    this._disposed = false;
    // Scene-parent tracking: when ALL tracked meshes are routed through
    // InstancedMesh slots, we detach `root` from its scene parent so three.js
    // stops walking it during updateMatrixWorld/render-list construction. The
    // instanced matrix in the per-asset InstancedMesh is the only state the
    // renderer needs. We remember the parent so we can re-attach when the
    // entity gets closer and needs its per-entity SkinnedMesh tree again.
    this._sceneParent = null;
    this._detached = false;
    // Ready promise resolves once the first LOD has been applied and the
    // root contains a renderable mesh.
    this.ready = this._bootstrap();
  }

  async _bootstrap() {
    try {
      await this.asset.ready;
      if (this._disposed) return;
      // Multi-driver VRM: @pixiv/three-vrm v3 exposes NO skeleton-rebind clone
      // (VRM.prototype = [constructor, update] only; no VRMUtils.clone), and its
      // humanoid/springBone/expression managers bind the bones of the scene they
      // were parsed against — so a _cloneSkinned copy cannot be driven by a
      // shared VRM runtime. Instead of sharing one runtime (which froze all-but-
      // one instance in bind pose), EACH entity re-parses its own independent
      // VRM from the asset's retained GLB bytes: its own scene, its own managers,
      // driven independently by vrm.update(dt) in _update(). This mirrors how a
      // multiplayer host parses one VRM per player.
      let cloned;
      if (this.asset.hasVRM && this.asset.rootBytes) {
        const own = await _parseOwnVrm(this.asset.rootBytes);
        if (this._disposed) { try { VRMUtils.deepDispose(own.scene); } catch (e) {} return; }
        this.vrm = own.vrm;
        cloned = own.scene;
      } else {
        // No VRM: render a skeleton-rebound clone sharing materials/geometry.
        cloned = _cloneSkinned(this.asset.rootGltf.scene);
        this.vrm = null;
      }
      this.root.add(cloned);
      // Compose world matrices so we can capture each mesh node's transform
      // RELATIVE to the entity root (root.matrixWorld⁻¹ × mesh.matrixWorld).
      // This relative is what the per-entity textured tier applies but the
      // instanced FAR tier historically dropped (it used bare root.matrixWorld),
      // causing the on-LOD-switch orientation flip. Captured once — the glTF
      // node hierarchy is static per entity.
      this.root.updateMatrixWorld(true);
      const _rootInv = new THREE.Matrix4().copy(this.root.matrixWorld).invert();
      // Cluster mode: replace each cloned mesh with a ClusterLodMesh sharing the
      // asset's prepared geometry. The per-cluster frustum-cull + LOD-select +
      // single multi-draw runs in ClusterLodMesh.onBeforeRender off the live
      // camera, so the entity needs NO per-frame LOD work (trackedMeshes stays
      // empty and _update's discrete-LOD pass is a no-op for this entity).
      if (this.asset.clusterMeshes && this.asset.clusterMeshes.length) {
        const meshOrderC = [];
        cloned.traverse((c) => { if (c.isMesh) meshOrderC.push(c); });
        this.clusterMeshes = [];
        for (let i = 0; i < this.asset.clusterMeshes.length; i++) {
          const cm = this.asset.clusterMeshes[i];
          const src = meshOrderC[i] || meshOrderC[0];
          if (!src) continue;
          const clm = new ClusterLodMesh(cm.geometry, cm.material, cm.clusterSet, {
            lod0Count: cm.lod0Count, // live viewport height read from renderer per frame
          });
          // Apply the source node's placement RELATIVE to the entity root, then
          // parent the clm under root directly -- NOT under src.parent. src.parent
          // is the glTF node whose own TRS (e.g. a 0.03 import scale + axis-fix
          // rotation) is ALREADY folded into src.matrixWorld; re-parenting the clm
          // under it applied that TRS a SECOND time (0.03*0.03 squared scale +
          // doubled rotation -> a tiny, mis-rotated model with the collider still
          // correct). This mirrors the discrete-LOD path below (relToRoot via
          // _rootInv) so cluster and discrete entities place identically.
          src.updateWorldMatrix(true, false);
          clm.applyMatrix4(new THREE.Matrix4().multiplyMatrices(_rootInv, src.matrixWorld));
          this.root.add(clm);
          src.removeFromParent();
          // Draw the FULL index range. ClusterLodMesh.onBeforeRender populates
          // geometry GROUPS (the per-cluster visible LOD sub-ranges) each frame and
          // lets three's normal pipeline draw them with the correct VAO; three only
          // draws groups that fall within drawRange, so drawRange must span the whole
          // index. (The old 3-index sentinel paired with the previous custom multi-draw
          // hook -- which is gone; a clamped drawRange now hides almost all geometry and
          // also breaks Mesh.raycast, which is bounded by drawRange.) The default full
          // range still fires onBeforeRender (three only skips it for a zero-count range).
          cm.geometry.setDrawRange(0, Infinity);
          this.clusterMeshes.push(clm);
        }
        this.emit('ready', this);
        return;
      }

      // Discover tracked meshes by descriptor.
      const meshOrder = [];
      cloned.traverse((c) => { if (c.isMesh) meshOrder.push(c); });
      for (let i = 0; i < this.asset.meshLodDescs.length; i++) {
        const desc = this.asset.meshLodDescs[i];
        const mesh = meshOrder[i] || meshOrder[0];
        if (!mesh) continue;
        const inlineLodIdx = desc.lods.findIndex((l) => l.inline);
        // Capture mesh-node transform relative to root; null when identity so
        // the per-frame slot-matrix path takes the cheap root-only fast path.
        mesh.updateWorldMatrix(true, false);
        const relToRoot = new THREE.Matrix4().multiplyMatrices(_rootInv, mesh.matrixWorld);
        const isRelIdentity = relToRoot.equals(_identityMatrix);
        this.trackedMeshes.push({
          meshDescIdx: i,
          currentLod: inlineLodIdx >= 0 ? inlineLodIdx : 0,
          mesh,
          _meshLocalToRoot: isRelIdentity ? null : relToRoot,
          baseIsSkinnedMesh: !!mesh.isSkinnedMesh,
          baseMaterial: mesh.material,
          baseSkeleton: mesh.skeleton || null,
          parent: mesh.parent,
          texState: this.asset.texLodDescs.map(() => ({ currentLod: 0 })),
          // Texture descriptors THIS mesh's material actually uses, by name.
          // Without this scoping the per-frame LOD-driven _applyTexLod loop ran
          // over ALL asset-wide descriptors (texState spans every texLodDesc),
          // so a foreign descriptor's bitmap got stamped into this mesh's base
          // map via the _findMaterialSlots fallback — the "texture changing on
          // multi-texture objects after LOD switch" bug. Scoping to the mesh's
          // own descriptors means foreign descriptors are never applied here.
          _texDescIdxs: _meshTexDescIdxs(mesh.material, this.asset.texLodDescs),
          vcMaterial: null,
          _instancedSlot: null,
          _instancedSlotIdx: -1,
          _instancedBoundRadius: null,
          _matrixNeedsUpdate: false,
          _precomputedTexLods: null, // Cached texture LODs for current mesh LOD
        });
      }
      // Animation: build mixer if source has clips.
      const animations = this.asset.rootGltf.animations || [];
      if (animations.length) {
        this.animationClips = animations;
        this.animationMixer = new THREE.AnimationMixer(cloned);
        const desiredIdx = Math.min(this.opts.animationIndex ?? 0, animations.length - 1);
        this.animationAction = this.animationMixer.clipAction(animations[desiredIdx]);
        this.animationAction.setLoop(THREE.LoopRepeat).play();
      }
      this.emit('ready', this);
    } catch (e) {
      this.emit('error', e);
    }
  }

  // World matrix the instanced slot must use for `tm`. Equals the mesh node's
  // world transform = root.matrixWorld × (mesh-local-relative-to-root). The
  // relative part is captured once at bootstrap (tm._meshLocalToRoot) because
  // the glTF node hierarchy is static per entity; only root.matrixWorld changes
  // per frame. Falls back to root.matrixWorld when the relative is identity.
  _slotWorldMatrix(tm) {
    const rel = tm._meshLocalToRoot;
    if (!rel) return this.root.matrixWorld;
    if (!this._slotMtx) this._slotMtx = new THREE.Matrix4();
    return this._slotMtx.multiplyMatrices(this.root.matrixWorld, rel);
  }

  // Update one tracked mesh to the desired LOD index. Materializes the
  // shell-type change (SkinnedMesh ↔ Mesh) and material swap as needed.
  async _applyLod(tm, wantIdx, screenPx = 0) {
    if (this._disposed) return;
    if (wantIdx === tm.currentLod) return;
    const desc = this.asset.meshLodDescs[tm.meshDescIdx];
    if (!desc) return;
    const target = desc.lods[wantIdx];
    if (!target) return;
    // Fetch geometry (cached or on-demand).
    let geo;
    if (target.inline) {
      geo = this.asset.geoCache.get(`${desc.meshIndex}:${desc.primIndex}:${wantIdx}`);
    } else {
      // Use the cached geometry immediately if we already have it (the common
      // case once an asset has streamed in). Only defer when it is genuinely not
      // yet resident — otherwise re-queuing an already-loaded LOD 0 on every
      // zoom-out made entities WAIT in the 2-wide deferred queue and pop back in
      // in waves ("disappear on LOD change, reappear all at once").
      const cachedGeo = this.asset.geoCache.get(`${desc.meshIndex}:${desc.primIndex}:${wantIdx}`);
      if (cachedGeo) {
        geo = cachedGeo;
      } else {
        // NOT resident. Register the want with the frame-budgeted warm loader
        // (network-lazy, GPU-eager, piecemeal) so we don't pay fetch+decode on
        // the switch frame, AND record _lodWantIdx so the picker doesn't re-fire
        // every frame for the same unresident target (that was the 11k churn).
        tm._lodWantIdx = wantIdx;
        this.pool._enqueueLodWarm(this.asset, tm.meshDescIdx, wantIdx, this._currentDistance);
        // BUT do not strand the entity at an expensive LOD: if we currently have
        // NO renderable geometry, or the target is a CHEAPER (lower) LOD than
        // current (e.g. dropping to the far/unskinned tier to recover FPS), fall
        // back to awaiting this load now — letting an entity stay stuck at a
        // costly LOD because a cheaper one isn't cached yet deadlocks FPS.
        const droppingToCheaper = wantIdx < tm.currentLod;
        const haveRenderable = tm.mesh && tm.mesh.geometry && tm.mesh.geometry.attributes.position;
        if (droppingToCheaper || !haveRenderable) {
          geo = await this.asset.ensureMeshLod(tm.meshDescIdx, wantIdx);
          if (this._disposed || !geo) return;
          tm._lodWantIdx = -1;
        } else {
          return; // promoting to a richer LOD can wait for the warm loader
        }
      }
    }
    if (this._disposed || !geo) return;
    if (wantIdx === tm.currentLod) return; // raced
    const kind = target.kind || 'textured';
    const wantSkinned = kind !== 'unskinned' && tm.baseIsSkinnedMesh;
    const haveSkinned = !!tm.mesh.isSkinnedMesh;

    // Instanced mode: when the target LOD is unskinned, route this entity
    // through a shared InstancedMesh slot instead of keeping its own mesh
    // tree. This collapses N entities into 1 draw call at the lowest LOD.
    const wantInstanced = kind === 'unskinned';
    const haveInstanced = tm._instancedSlot != null;
    if (wantInstanced) {
      const slot = this.pool._getInstancedSlot(this.asset, tm.meshDescIdx, wantIdx);
      if (slot) {
        // Hide the entity's own mesh from the renderer.
        tm.mesh.visible = false;
        // Acquire a slot index (re-acquire if changing LOD within unskinned tier).
        if (haveInstanced && (tm._instancedSlot !== slot)) {
          tm._instancedSlot.releaseSlot(this);
        }
        if (!haveInstanced || tm._instancedSlot !== slot) {
          tm._instancedSlot = slot;
          tm._instancedSlotIdx = slot.acquireSlot(this);
          tm._matrixNeedsUpdate = true;
        }
        // Seed the per-instance world-space bound sphere for GPU culling.
        // Recomputed once here (entity transform is stable for typical
        // static spawns; movers refresh it in _update). For animated
        // entities this is fine — far-tier instanced LODs are unskinned
        // bind-pose, so the sphere envelope is constant.
        {
          this.root.updateMatrixWorld(true);
          const worldMat = this._slotWorldMatrix(tm);
          // Write the instance transform IMMEDIATELY at slot acquisition. This
          // is essential for the GPU data-texture path: _applyLod is async (it
          // awaits geometry load), so the _update() call that triggered it has
          // already run its later matrix-write block by the time we get here —
          // the write would otherwise be deferred to next frame, and if the
          // entity is static + camera still, the static fast-skip returns early
          // and the texel never gets written -> instance stays at the zero
          // matrix -> invisible (the "lowest LODs disappear" bug). Writing here
          // makes the texel correct the instant the slot is acquired.
          slot.setMatrixForSlot(tm._instancedSlotIdx, worldMat);
          tm._matrixNeedsUpdate = false;
          const sphere = geo.boundingSphere;
          if (sphere) {
            const scale = _maxAbsScale(this.root.scale);
            tm._instancedBoundRadius = sphere.radius;
            slot.setBoundSphereForSlot(tm._instancedSlotIdx, sphere.radius * scale);
          }
        }
        tm.currentLod = wantIdx;
        // Pre-compute texture LODs for this mesh LOD at the current screen-pixel size.
        tm._precomputedTexLods = _precomputeAllTexLods(this.asset, screenPx);
        this.emit('lod-changed', { entity: this, meshDescIdx: tm.meshDescIdx, lod: wantIdx, kind, instanced: true });
        return;
      }
      // Slot unavailable (geo not yet loaded for instancing) → fall through to non-instanced path.
    } else if (haveInstanced) {
      // Leaving the instanced tier — release the slot and re-show our own mesh.
      tm._instancedSlot.releaseSlot(this);
      tm._instancedSlot = null;
      tm._instancedSlotIdx = -1;
      tm.mesh.visible = true;
    }
    // Material selection per kind.
    // MATERIAL GROUPING OPTIMIZATION: Use global tier materials if enabled
    let mat;
    if (kind === 'textured') {
      // Textured (HERO/MID) tier MUST use the entity's OWN baseMaterial, which
      // carries that asset's textures (map/normalMap/...). The shared global
      // tier material cannot hold N different per-asset textures, so routing
      // textured LODs through it rendered them WHITE (only vertex colors showed).
      // _applyTexLod swaps texture LODs on tm.baseMaterial, gated by
      // `mat === tm.baseMaterial` — so the base material is also required for
      // textures to update at all. (Global pool grouping still applies to the
      // FAR/vertex-color tier where it's correct.)
      mat = tm.baseMaterial;
    } else {
      if (!tm.vcMaterial) {
        const m = new THREE.MeshLambertMaterial({ vertexColors: true });
        m.onBeforeCompile = (shader) => {
          // sRGB->linear decode per-VERTEX (not per-fragment): see batched-far-tier.
          shader.vertexShader = shader.vertexShader.replace(
            '#include <color_vertex>',
            `#include <color_vertex>
            #if defined( USE_COLOR_ALPHA )
              vColor.rgb = pow(vColor.rgb, vec3(2.2));
            #elif defined( USE_COLOR )
              vColor = pow(vColor, vec3(2.2));
            #endif`
          );
        };
        tm.vcMaterial = m;
      }
      mat = tm.vcMaterial;
    }
    if (wantSkinned === haveSkinned) {
      tm.mesh.geometry = geo;
      tm.mesh.material = mat;
    } else {
      const parent = tm.mesh.parent || tm.parent;
      let next;
      if (wantSkinned) {
        next = new THREE.SkinnedMesh(geo, mat);
        if (tm.baseSkeleton) next.bind(tm.baseSkeleton);
      } else {
        next = new THREE.Mesh(geo, mat);
      }
      next.frustumCulled = false;
      // Copy local transform. LODs may have different mesh-local transforms
      // due to baking differences, but since mesh is a direct child of root,
      // we use the same relative positioning.
      next.position.copy(tm.mesh.position);
      next.quaternion.copy(tm.mesh.quaternion);
      next.scale.copy(tm.mesh.scale);
      next.name = tm.mesh.name;
      if (parent) {
        parent.remove(tm.mesh);
        parent.add(next);
      }
      tm.mesh = next;
    }
    tm.currentLod = wantIdx;
    // Pre-compute texture LODs for this mesh LOD at the current screen-pixel size.
    tm._precomputedTexLods = _precomputeAllTexLods(this.asset, screenPx);
    this.emit('lod-changed', { entity: this, meshDescIdx: tm.meshDescIdx, lod: wantIdx, kind });
  }

  async _applyTexLod(tm, tdIdx, wantIdx) {
    if (this._disposed) return;
    const tState = tm.texState[tdIdx];
    if (!tState || wantIdx === tState.currentLod) return;
    const desc = this.asset.texLodDescs[tdIdx];
    if (!desc) return;
    let bmp;
    const target = desc.lods[wantIdx];
    if (target.inline) {
      bmp = this.asset.texCache.get(`${desc.textureIndex}:${wantIdx}`);
    } else {
      bmp = await this.asset.ensureTexLod(tdIdx, wantIdx);
    }
    if (this._disposed || !bmp) return;
    if (wantIdx === tState.currentLod) return;
    // Apply to matching slot(s) on tm.mesh.material — but ONLY when the
    // current material is the textured baseMaterial. Vertex-color LODs
    // ignore textures.
    const mat = tm.mesh.material;
    if (mat === tm.baseMaterial) {
      const targets = _findMaterialSlots(mat, desc);
      // Far texture LODs (idx>=3) get cheap linear/no-mipmap filtering — at that
      // distance the mip chain isn't worth the upload/bandwidth. These run on the
      // per-entity baseMaterial (not the shared pool material), so it's safe.
      const farLod = wantIdx >= 3;
      // Anisotropic filtering on the close, mipmapped tiers (wantIdx < 3) keeps
      // oblique surfaces sharp at near-zero cost on a draws~0 GPU-bound scene.
      // Cached once off the renderer; capped at 8 so we never overpay on GPUs
      // that advertise 16x. Far LODs skip it (no mipmaps, linear-only).
      const aniso = this.pool._maxAnisotropy ??
        (this.pool._maxAnisotropy = Math.min(8, this.pool.renderer?.capabilities?.getMaxAnisotropy?.() ?? 1));
      for (const tex of targets) {
        tex.dispose();
        tex.image = bmp;
        if (farLod) {
          tex.minFilter = THREE.LinearFilter;
          tex.magFilter = THREE.LinearFilter;
          tex.generateMipmaps = false;
          tex.anisotropy = 1;
        } else {
          tex.anisotropy = aniso;
        }
        tex.needsUpdate = true;
      }
    }
    tState.currentLod = wantIdx;
  }

  // Detach our root from the scene graph when every tracked mesh is routed
  // through an InstancedMesh slot. three.js then skips this whole subtree
  // during updateMatrixWorld and render-list construction. The instanced-mesh
  // matrix is the only renderer-visible state we still need to push.
  _maybeDetach() {
    if (this._detached || this._disposed) return;
    if (!this.trackedMeshes.length) return;
    for (const tm of this.trackedMeshes) {
      if (!tm._instancedSlot) {
        this._allInstanced = false;
        return; // at least one mesh still needs per-entity draw
      }
    }
    this._allInstanced = true;
    const parent = this.root.parent;
    if (!parent) return;
    this._sceneParent = parent;
    parent.remove(this.root);
    this._detached = true;
  }
  _maybeReattach() {
    if (!this._detached || this._disposed) return;
    // Re-attach as soon as ANY tracked mesh leaves the instanced tier.
    let allInstanced = true;
    for (const tm of this.trackedMeshes) {
      if (!tm._instancedSlot) { allInstanced = false; break; }
    }
    if (allInstanced) return;
    this._allInstanced = false;
    if (this._sceneParent) {
      this._sceneParent.add(this.root);
      // Force a matrixWorld recompute on next frame since the subtree was
      // detached and possibly skipped updates.
      this.root.matrixWorldNeedsUpdate = true;
    }
    this._detached = false;
  }

  // Per-frame update called by the pool. Returns {distance, screenPx, tier} for tier allocation.
  // Used to track per-frame budget consumption across HERO/MID/FAR tiers.
  _update(camera, viewportHeight, dt, globalCeilingLod, frustum, animationThrottleDistance) {
    if (this._disposed) return { distance: Infinity, screenPx: 0, tier: 'far' };
    // Cluster entities self-drive LOD in ClusterLodMesh.onBeforeRender; the entity
    // does no per-frame LOD work. Keep matrices fresh and return a neutral result.
    if (this.clusterMeshes && this.clusterMeshes.length) {
      if (this.root.matrixAutoUpdate) this.root.updateMatrixWorld();
      if (this.animationMixer) this.animationMixer.update(dt);
      return { distance: 0, screenPx: 9999, tier: 'cluster' };
    }
    let primaryMesh = this.trackedMeshes[0]?.mesh;
    if (!primaryMesh) return { distance: Infinity, screenPx: 0, tier: 'far' };
    // FAST SKIP (static + camera still + fully GPU-instanced): nothing this
    // entity contributes can change this frame. Its world matrix is already
    // uploaded to the instance buffer, the GPU vertex shader culls it against
    // the per-frame frustum uniform, and its LOD can't change without camera
    // motion (distance is constant). So we return the cached result without
    // touching distance/frustum/LOD/matrix math. This is the bulk of the
    // ~25ms/frame JS entity-loop cost at 500 static entities — it should fall
    // toward zero for a still camera. Any of: entity movable, camera moved this
    // frame, not-yet-fully-instanced, ceiling changed, or no cached result yet
    // forces the full path. (globalCeilingLod change must re-pick LODs.)
    const _movable = this.root.matrixAutoUpdate || this._boundDirty;
    if (!_movable && !this.pool._cameraMoved && this._allInstanced && this._lastUpdateResult
        && this._lastCeilingLod === globalCeilingLod) {
      return this._lastUpdateResult;
    }
    // Remember scene parent the first time we see it so we can re-attach
    // later even if the root was detached for being fully instanced.
    if (!this._sceneParent && this.root.parent) this._sceneParent = this.root.parent;
    // Lazy matrix update: only invalidated when root.position/rotation/scale
    // changed since last tick. The autoUpdate flag controls this.
    // When detached, this is a no-op walk over an unparented root — still
    // cheap, and keeps matrixWorld current for instance-slot writes.
    if (this.root.matrixWorldNeedsUpdate || this.root.matrixAutoUpdate) {
      this.root.updateMatrixWorld(true);
    }
    const sphere = primaryMesh.geometry?.boundingSphere;
    if (!sphere) return { distance: Infinity, screenPx: 0, tier: 'far' };
    const world = _tmpV3.setFromMatrixPosition(primaryMesh.matrixWorld);
    // (Removed an unused `world.distanceTo(camera.position)` here — it computed
    // a sqrt for every entity every frame and its result was never read; the
    // distance actually used is computed once below, after the frustum test.)
    const scaleLen = _maxAbsScale(this.root.scale);
    const radius = sphere.radius * scaleLen;
    // Skip frustum check + LOD if we're MUCH closer than necessary OR much
    // farther than we can resolve. The frustum test itself is moderately
    // expensive (matrix-vs-sphere per plane).
    // If any tracked mesh is currently in an instanced slot, the GPU
    // vertex shader handles frustum culling for us — skip the CPU sphere
    // test entirely. Per-entity meshes (HERO/MID) still need it.
    // Optimize: test frustum using dynamic interval based on scene staticness.
    // Static entities: 8-10 frame interval. Moving entities: check every frame (interval = 0).
    const movable = this.root.matrixAutoUpdate || this._boundDirty;
    let inFrustum = this._lastInFrustum; // Use cached result by default
    if (this._allInstanced) {
      inFrustum = true;
    } else if (this.pool._enableFrustumCulling) {
      // For moving entities, always test (interval = 0).
      // For static entities, use pool's dynamic interval (5/8/10 depending on scene staticness).
      const effectiveInterval = movable ? 0 : this.pool._dynamicFrustumCheckInterval;
      if (this._firstFrustumTest || this._frustumCheckInterval <= 0 || movable) {
        // Test frustum if first update, interval expired, or entity is moving
        _tmpSphere.set(world, radius);
        inFrustum = frustum ? frustum.intersectsSphere(_tmpSphere) : true;
        this._frustumCheckInterval = effectiveInterval;
        this._lastInFrustum = inFrustum;
        this._cachedFrustumVisible = inFrustum; // Update visibility cache
        this._firstFrustumTest = false;
      } else {
        // Use cached frustum result, just decrement interval
        this._frustumCheckInterval--;
      }
    } else {
      // Frustum culling disabled: always show
      inFrustum = true;
    }
    if (this.root.visible !== inFrustum) this.root.visible = inFrustum;
    if (!inFrustum) {
      // OPTIMIZATION: For static entities that are out-of-frustum, skip expensive
      // per-frame work (matrix writes, LOD selection, screenPx calculation, animation).
      // Only movable entities (moving or flagged for matrix update) need matrix refresh.
      // Static out-of-frustum entities can skip nearly all work until they move or
      // camera swings back (next interval test).
      if (!movable) {
        // EARLY RETURN: Static entity out-of-frustum → skip everything
        // Just update interval counter and return (no matrix, no LOD, no animation)
        this._cachedFrustumVisible = false;
        this._maybeReattach();
        this._maybeDetach();
        return { distance: Infinity, screenPx: 0, tier: 'far' };
      }
      // Movable entity out-of-frustum: still push instance matrix in case a slot
      // is active — we want the matrix valid when the camera swings back. Note: we
      // zero it via visible:false on the root, but the InstancedMesh ignores root
      // visibility (it's a sibling in the scene tree). Set the slot matrix to a
      // far-away point so it's outside the camera regardless.
      for (const tm of this.trackedMeshes) {
        if (tm._instancedSlot && tm._instancedSlotIdx >= 0 && (movable || tm._matrixNeedsUpdate)) {
          // Use the MESH NODE's world matrix, not the bare root matrix. The
          // glTF mesh node may carry a non-identity local transform inside the
          // cloned scene hierarchy (root -> ...cloned chain... -> mesh). The
          // per-entity textured tier renders through that full chain, so the
          // instanced FAR tier must compose the same matrix or the model FLIPS
          // orientation/position on every tier switch (see _slotWorldMatrix).
          tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, this._slotWorldMatrix(tm));
          tm._matrixNeedsUpdate = false;
        }
      }
      this._maybeReattach();
      this._maybeDetach();
      return { distance: Infinity, screenPx: 0, tier: 'far' };
    }
    const dist = camera.position.distanceTo(world);
    // GPU-optimized LOD selection: screenPx calculation is now done in the
    // vertex shader (_patchInstancedSlotMaterial), reducing CPU overhead from
    // ~1.2µs per entity (1000 entities = 1.2ms) to ~0.1µs (frustum test only).
    // LOD decisions are made per-instance in the GPU, passed via vLodIndex.
    // CPU-side LOD updates only happen when distance crosses significant thresholds.
    // For non-instanced entities (HERO/MID tiers), we still need estimated screenPx
    // for tier allocation, but we can use a cached/estimated value that updates
    // only when the entity actually moves.
    let screenPx = 0;
    // Recompute screenPx when the entity moved OR the CAMERA moved this frame.
    // Static entities (matrixAutoUpdate=false) used to freeze _lastScreenPx at
    // spawn distance, so their LOD never changed as the camera orbited/zoomed.
    // pool._cameraMoved is set once per frame in update().
    if (this._lastScreenPx != null && !movable && !this.pool._cameraMoved) {
      // Use cached screenPx (entity static AND camera static this frame).
      screenPx = this._lastScreenPx;
    } else {
      // Compute screenPx only for moving entities or first update.
      // fovTanHalf = tan(fov/2) is camera-constant for the frame; the pool
      // computes it once per update() (this.pool._fovTanHalf) so we avoid a
      // per-entity degToRad + tan (a transcendental call) every frame.
      const fovTanHalf = this.pool._fovTanHalf || Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      const halfWorld = fovTanHalf * (dist > 0.0001 ? dist : 0.0001);
      screenPx = (radius / halfWorld) * viewportHeight;
      this._lastScreenPx = screenPx;
    }
    // Tier routing: HERO/MID/FAR are all decided by which LOD the screen-
    // space picker selects. FAR-distance entities resolve to the unskinned
    // (idx 0) LOD which the pool routes through a shared per-asset
    // InstancedMesh — real 3D geometry, just decimated, so shape survives
    // for non-character meshes (terrain chunks, props, anything).
    const tinyOnScreen = screenPx < 4;
    // SUB-PIXEL CULL: below ~2px a model occupies <~4 screen pixels — not worth
    // rasterizing at all. Skip drawing it entirely (zero its instanced slot
    // matrix so the shared InstancedMesh draws nothing for it; hide a per-entity
    // mesh). This removes vertex+fill work for the densest part of a big scene
    // (most models are tiny dots at a far camera). Restores the instant it grows
    // back above the threshold. Hysteresis (2px cull / 3px restore) avoids
    // flicker at the boundary.
    // Threshold is tunable (pool._subPixelCullPx, default 2px) with +1px restore
    // hysteresis. Default is conservative so only truly invisible dots are cut;
    // raise it to trade a little far-detail for FPS in dense scenes.
    const base = this.pool._subPixelCullPx ?? 2;
    const cullPx = this._subPixelCulled ? base + 1 : base;
    const wantSubPixelCull = screenPx > 0 && screenPx < cullPx;
    if (wantSubPixelCull !== this._subPixelCulled) {
      this._subPixelCulled = wantSubPixelCull;
      for (const tm of this.trackedMeshes) {
        if (tm._instancedSlot && tm._instancedSlotIdx >= 0) {
          tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, wantSubPixelCull ? _zeroMatrix : this._slotWorldMatrix(tm));
        } else if (tm.mesh) {
          tm.mesh.visible = !wantSubPixelCull;
        }
      }
    }
    // Octahedral impostor FINAL LOD: in the far band (above sub-pixel cull,
    // below _impostorPx) draw the WHOLE entity as one billboard and skip the
    // per-primitive LOD/matrix work below. Entry/exit hides/restores the
    // entity's normal draws (respecting any sub-pixel cull).
    let impostorHandled = false;
    if (this.pool._useImpostorFinalLod) {
      impostorHandled = (!this._subPixelCulled)
        ? this._updateImpostor(screenPx, world, radius, movable)
        : (this._impostorActive ? (this._exitImpostor(), false) : false);
    }
    if (!impostorHandled) {
      // Re-pick LOD only when something that affects the choice changed since
      // this entity last picked: the pool bumps _lodEpoch when the camera moved
      // or _lodDistanceScale shifted beyond a quantum. A still camera + stable
      // scale => epoch unchanged => no re-pick, no _applyLod churn. This is the
      // core stutter fix: previously every visible textured-tier entity re-ran
      // the picker every frame, and any boundary jitter (scale-hunting) flipped
      // it, producing hundreds of switches/sec even with a still camera.
      const lodEpochChanged = this._lodPickEpoch !== this.pool._lodEpoch;
      if (lodEpochChanged) this._lodPickEpoch = this.pool._lodEpoch;
      if (!tinyOnScreen && !this._subPixelCulled && lodEpochChanged) {
        for (const tm of this.trackedMeshes) {
          const desc = this.asset.meshLodDescs[tm.meshDescIdx];
          if (!desc) continue;
          // Pick the LOD for the current screen size — ALSO for entities already
          // in an instanced slot. Previously this was gated behind
          // `if (!tm._instancedSlot)` with the (false) assumption the GPU does
          // LOD selection; the GPU only culls. That left instanced/FAR entities
          // STUCK at their entry LOD forever, so LOD never changed as the camera
          // moved closer. _applyLod handles the instanced<->per-entity transition
          // (wantInstanced/haveInstanced), so calling it here lets a FAR entity
          // promote to a higher textured LOD when the camera approaches and demote
          // back when it recedes.
          const targetIdx = _pickMeshLod(desc.lods, screenPx, globalCeilingLod, this.pool._use3LodSystem, this.pool._lodDistanceScale, tm.currentLod);
          // HYSTERESIS + IN-FLIGHT GUARD. Without these the entity re-fires
          // _applyLod every frame it sits near a LOD threshold (the picker has no
          // dead-band) AND re-fires every frame while an async ensureMeshLod is
          // still loading (the deferred path returns without setting currentLod),
          // producing tens of thousands of switches and continuous stutter as
          // the camera moves across the vertcolor<->textured band. Fix: only
          // commit a switch after the SAME target has been requested for a few
          // consecutive evaluations (debounce ping-pong), and never re-issue
          // while a switch is already in flight.
          // If we've already registered a want for this exact target and its geo
          // is still not resident, don't re-invoke _applyLod — the warm loader
          // owns it. This stops the per-frame cache-miss re-fire (was 11k/dolly).
          const wantKey = `${desc.meshIndex}:${desc.primIndex}:${targetIdx}`;
          const targetResident = desc.lods[targetIdx] && (desc.lods[targetIdx].inline || this.asset.geoCache.has(wantKey));
          if (tm._lodWantIdx === targetIdx && !targetResident) {
            // pending in the warm loader — leave it; it'll apply when resident.
          } else if (targetIdx !== tm.currentLod && !tm._lodPending) {
            if (tm._pendingLodTarget === targetIdx) {
              tm._lodConfirm = (tm._lodConfirm || 0) + 1;
            } else {
              tm._pendingLodTarget = targetIdx;
              tm._lodConfirm = 1;
            }
            const needed = this.pool._lodSwitchConfirmFrames ?? 4;
            if (tm._lodConfirm >= needed) {
              tm._lodConfirm = 0;
              tm._pendingLodTarget = -1;
              tm._lodPending = true;
              const px = screenPx;
              this._applyLod(tm, targetIdx, px).finally(() => { tm._lodPending = false; });
            }
          } else if (targetIdx === tm.currentLod) {
            tm._lodWantIdx = -1;
            // Settled at target — reset the debounce so a future change starts fresh.
            tm._pendingLodTarget = -1; tm._lodConfirm = 0;
          }
          // Use pre-computed texture LODs (only meaningful for the non-instanced,
          // textured tiers; instanced FAR uses vertex color, no textures).
          if (!tm._instancedSlot && this.pool._enableTextureLod && tm._precomputedTexLods) {
            // Only this mesh's OWN texture descriptors — never foreign ones (see
            // _texDescIdxs). Falls back to every descriptor when the per-mesh
            // mapping came back empty (no name match), preserving prior behaviour
            // for assets whose texture names don't resolve.
            const idxs = (tm._texDescIdxs && tm._texDescIdxs.length)
              ? tm._texDescIdxs
              : tm.texState.map((_, i) => i);
            for (const ti of idxs) {
              const tWant = tm._precomputedTexLods[ti];
              if (tWant != null && tWant !== tm.texState[ti].currentLod) {
                this._applyTexLod(tm, ti, tWant);
              }
            }
          }
        }
      }
    }
    // Push instance matrices for instanced-tier tracked meshes. Also
    // refresh the per-instance world-space bound-sphere center for GPU
    // frustum culling — only when the entity actually moves (root has
    // auto-update on, or is flagged for one-shot rebuild).
    // Skipped while the impostor tier owns this entity (its tracked draws are
    // parked at the zero matrix; pushing real transforms would un-hide them).
    for (const tm of (impostorHandled ? [] : this.trackedMeshes)) {
      if (tm._instancedSlot && tm._instancedSlotIdx >= 0) {
        // Compute the slot's world matrix at most ONCE per frame: both the matrix
        // push and the bound-sphere refresh below read the same transform, and
        // _slotWorldMatrix does a matrix multiply (and, for non-fully-instanced
        // meshes, an allocation) on every call. Cache it lazily here.
        let slotWM = null;
        // Sub-pixel-culled entities keep their zeroed slot matrix — don't push
        // the real transform back (that would un-cull them every frame).
        if (!this._subPixelCulled && (movable || tm._matrixNeedsUpdate)) {
          slotWM = this._slotWorldMatrix(tm);
          tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, slotWM);
          tm._matrixNeedsUpdate = false;
        }
        if (!this._subPixelCulled && movable && tm._instancedBoundRadius != null) {
          const scale = _maxAbsScale(this.root.scale);
          tm._instancedSlot.setBoundSphereForSlot(
            tm._instancedSlotIdx,
            tm._instancedBoundRadius * scale,
          );
        }
      }
    }
    if (movable) {
      // Invalidate cached screenPx when entity moves so it gets recalculated next frame
      this._lastScreenPx = null;
    }
    this._boundDirty = false;
    // If every tracked mesh is now instanced, detach root from the scene
    // graph so three.js stops paying traversal cost on it. Re-attach as soon
    // as any tracked mesh leaves the instanced tier.
    this._maybeReattach();
    this._maybeDetach();
    // Optimization 3: Aggressive animation throttling based on distance tiers
    // <10m: Full animation (1x time-skip)
    // 10-20m: Half-rate (every 2 frames, 2x skip)
    // 20-40m: Quarter-rate (every 4 frames, 4x skip)
    // >40m: Disabled (bind pose only)
    // Skip entirely when the entity is instanced (bind pose only).
    // Only probe instanced state when there's actually an animation consumer —
    // entities with no mixer and no VRM (the common FAR-tier case) skip the
    // trackedMeshes walk entirely.
    const hasAnimConsumer = !!this.animationMixer || !!this.vrm?.update;
    const anyInstanced = hasAnimConsumer
      ? (this._allInstanced || this.trackedMeshes.some((tm) => !!tm._instancedSlot))
      : false;
    if (this.animationMixer && !anyInstanced) {
      if (this.pool._enableAnimThrottle) {
        if (dist > 40) {
          // >40m: skip animation entirely (bind pose only)
          // no-op: don't call mixer.update()
        } else if (dist > 20) {
          // 20-40m: quarter-rate (every 4 frames)
          if ((this._animTickCounter = ((this._animTickCounter || 0) + 1)) % 4 === 0) {
            this.animationMixer.update(dt * 4);
          }
        } else if (dist > 10) {
          // 10-20m: half-rate (every 2 frames)
          if ((this._animTickCounter = ((this._animTickCounter || 0) + 1)) % 2 === 0) {
            this.animationMixer.update(dt * 2);
          }
        } else {
          // <10m: full animation (1x)
          this.animationMixer.update(dt);
        }
      } else {
        // Throttling disabled: always full rate
        this.animationMixer.update(dt);
      }
    }
    // opts.driveVrm === false lets the HOST own vrm.update(dt) — e.g. a game
    // that runs its own animator/expression/look-at pipeline on entity.vrm and
    // would otherwise double-drive it. When the host drives, the pool also skips
    // the close-range throttle gate (the host decides its own update cadence).
    if (this.vrm?.update && this.opts.driveVrm !== false && dist < animationThrottleDistance && !anyInstanced) this.vrm.update(dt);
    // Return distance and screenPx for tier allocation by the pool. The result
    // is read synchronously by the pool before the next _update call, so a
    // single reused object avoids a per-entity allocation in the hot path.
    this._currentDistance = dist;
    const r = this._updateResult || (this._updateResult = { distance: 0, screenPx: 0, tier: 'unassigned' });
    r.distance = dist; r.screenPx = screenPx; r.tier = 'unassigned';
    // Cache for the static-fast-skip path at the top of _update. Safe to return
    // this same reused object next frame because a static entity's distance and
    // screenPx are unchanged while the camera is still. _lastCeilingLod guards
    // against a ceiling change forcing a re-pick.
    this._lastUpdateResult = r;
    this._lastCeilingLod = globalCeilingLod;
    return r;
  }

  // FINAL-LOD octahedral impostor. Returns true if this entity is currently
  // represented by an impostor billboard (and the caller should skip the normal
  // per-primitive LOD + matrix work this frame). `world` is the primary mesh
  // world position, `worldRadius` its world bound radius, both already computed.
  _updateImpostor(screenPx, world, worldRadius, movable) {
    const tier = this.pool._getImpostorTier();
    if (!tier) return false;
    const onPx = this.pool._impostorPx ?? 14;
    const enterPx = this._impostorActive ? onPx + 2 : onPx; // +2 exit hysteresis
    if (!(screenPx > 0 && screenPx < enterPx)) {
      if (this._impostorActive) this._exitImpostor();
      return false;
    }
    // Use the asset's atlas if already baked; otherwise queue a bake (drained
    // 1/frame in update(), so a far camera bringing many assets into range at
    // once spreads the GRIDxGRID renders across frames instead of stuttering) and
    // stay on the current far LOD until the atlas is ready.
    const desc = this.pool._impostorTier && this.pool._impostorTier.hasAsset(this.asset)
      ? this.pool._impostorTier._assetLayers.get(this.asset.url)
      : null;
    if (!desc) {
      if (this._impostorActive) this._exitImpostor();
      this.pool._queueImpostorBake(this.asset, this);
      return false;
    }
    // Compose the world billboard placement from the asset-local descriptor and
    // this entity's transform (bounding-safe scale factor = max(|sx|,|sy|,|sz|),
    // tight and correct for both uniform and non-uniform scale). Only needed on
    // first acquire (static placement, computed once) or every frame while
    // movable — skip the applyMatrix4 + scale.length() sqrt entirely for a
    // static impostor that's already active.
    if (!this._impostorActive) {
      const wc = _tmpV3b.copy(desc.center).applyMatrix4(this.root.matrixWorld);
      const s = _maxAbsScale(this.root.scale);
      const wr = desc.radius * s;
      this._setTrackedDrawsHidden(true);
      this._impostorActive = true;
      this.pool._impostorActiveCount++;
      this._impostorId = tier.acquire(this, desc.layer, wc.x, wc.y, wc.z, wr);
    } else if (movable) {
      const wc = _tmpV3b.copy(desc.center).applyMatrix4(this.root.matrixWorld);
      const s = _maxAbsScale(this.root.scale);
      const wr = desc.radius * s;
      tier.setCenter(this._impostorId, wc.x, wc.y, wc.z, wr);
    }
    return this._impostorId >= 0;
  }

  _exitImpostor() {
    if (!this._impostorActive) return;
    if (this.pool._impostorTier) this.pool._impostorTier.release(this);
    this._impostorActive = false;
    this._impostorId = -1;
    this.pool._impostorActiveCount = Math.max(0, this.pool._impostorActiveCount - 1);
    this._setTrackedDrawsHidden(false); // respects sub-pixel cull internally
  }

  // Hide (or restore) every tracked-mesh draw for this entity. Mirrors the
  // sub-pixel cull pattern: instanced slots park at the zero matrix, per-entity
  // meshes toggle visibility. On restore, a still-sub-pixel-culled entity stays
  // hidden so impostor exit can't accidentally un-cull it.
  _setTrackedDrawsHidden(hidden) {
    const cull = hidden || this._subPixelCulled;
    for (const tm of this.trackedMeshes) {
      if (tm._instancedSlot && tm._instancedSlotIdx >= 0) {
        tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, cull ? _zeroMatrix : this._slotWorldMatrix(tm));
      } else if (tm.mesh) {
        tm.mesh.visible = !cull;
      }
    }
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    if (this._impostorActive && this.pool._impostorTier) this.pool._impostorTier.release(this);
    this.root.parent?.remove(this.root);
    // If we were detached for being fully instanced, the root has no parent
    // but _sceneParent still holds the original — nothing to remove there
    // (we already removed ourselves at detach time). Clear the reference.
    this._sceneParent = null;
    this._detached = false;
    // Stop animation.
    if (this.animationAction) this.animationAction.stop();
    this.animationMixer = null;
    this.animationAction = null;
    // Release this entity's OWN VRM runtime. Each driven instance parsed its
    // own scene + managers (multi-driver), so deepDispose frees this instance's
    // springbone/collider/expression GPU resources without touching siblings or
    // the shared asset payload.
    if (this.vrm) {
      try { VRMUtils.deepDispose(this.vrm.scene); } catch (e) { /* best-effort */ }
    }
    this.vrm = null;
    // Release any instanced-mesh slots we held.
    for (const tm of this.trackedMeshes) {
      if (tm._instancedSlot) tm._instancedSlot.releaseSlot(this);
      if (tm.vcMaterial) tm.vcMaterial.dispose();
    }
    this.trackedMeshes = [];
    if (this.pool._occlusionTier) this.pool._occlusionTier.release(this);
    this.pool._entities.delete(this);
    this.emit('disposed', this);
  }
}

// Helper: clone a three.js scene with SkinnedMesh skeletons re-bound to the
// cloned bone tree. Crucial: MATERIALS AND GEOMETRIES are shared with the
// source — only the Object3D scene-graph topology and per-mesh skeleton
// objects are unique per entity. Without this every spawned entity allocates
// its own Texture/Material clones, leaking 100s of GPU textures with N=500
// even though they're never used.
function _cloneSkinned(source) {
  const sourceToClone = new Map();
  const cloneRoot = _cloneObject3D(source, sourceToClone);
  // For every SkinnedMesh, build a fresh skeleton with cloned bones.
  cloneRoot.traverse((cm) => {
    if (!cm.isSkinnedMesh) return;
    let sourceSm = null;
    for (const [s, c] of sourceToClone) {
      if (c === cm) { sourceSm = s; break; }
    }
    if (!sourceSm) return;
    const srcSkel = sourceSm.skeleton;
    if (!srcSkel) return;
    const newBones = srcSkel.bones.map((b) => sourceToClone.get(b) || b);
    const newSkel = new THREE.Skeleton(newBones, srcSkel.boneInverses);
    cm.bind(newSkel, cm.bindMatrix);
  });
  return cloneRoot;
}

// Selective clone: preserves shared materials/geometries, copies transforms.
function _cloneObject3D(src, sourceToClone) {
  let copy;
  if (src.isSkinnedMesh) {
    // Share geometry + material; per-entity skeleton is rebuilt above.
    copy = new THREE.SkinnedMesh(src.geometry, src.material);
    copy.bindMode = src.bindMode;
    copy.bindMatrix.copy(src.bindMatrix);
    copy.bindMatrixInverse.copy(src.bindMatrixInverse);
  } else if (src.isMesh) {
    copy = new THREE.Mesh(src.geometry, src.material);
  } else if (src.isBone) {
    copy = new THREE.Bone();
  } else {
    copy = new THREE.Object3D();
  }
  copy.name = src.name;
  copy.position.copy(src.position);
  copy.quaternion.copy(src.quaternion);
  copy.scale.copy(src.scale);
  copy.matrixAutoUpdate = src.matrixAutoUpdate;
  copy.visible = src.visible;
  copy.frustumCulled = src.frustumCulled;
  sourceToClone.set(src, copy);
  for (const child of src.children) {
    copy.add(_cloneObject3D(child, sourceToClone));
  }
  return copy;
}

// LOD picker — same logic as the inline demo, plus a ceiling clamp.
// For 3-LOD system: uses thresholds [50px, 25px, 10px] for LODs [0, 2, 4]
// For 5-LOD system: uses thresholds [80, 200, 400, 800, 1400] for LODs [0, 1, 2, 3, 4]
// Hoisted out of _pickMeshLod: these ladders are read-only (indexed/indexOf'd,
// never mutated) so allocating them fresh on every call (this runs per tracked
// mesh per frame) was pure per-call garbage. Module-level constants instead.
const _LOD_THRESHOLDS_3 = [50, 25, 10]; // Adjusted for 3-LOD: LOD0@50px, LOD2@25px, LOD4@10px
const _LOD_INDICES_3 = [0, 2, 4];
const _LOD_THRESHOLDS_5 = [80, 200, 400, 800, 1400];
const _LOD_INDICES_5 = [0, 1, 2, 3, 4];

function _pickMeshLod(lods, screenPx, ceilingIdx, use3LodSystem = false, lodScale = 1, curLod = -1) {
  let thresholds, lodIndices;
  if (use3LodSystem && lods.length >= 5) {
    // 3-LOD system: only use LODs [0, 2, 4] with thresholds [50px, 25px, 10px]
    // This skips intermediate LODs 1 and 3, saving VRAM and reducing memory churn
    thresholds = _LOD_THRESHOLDS_3;
    lodIndices = _LOD_INDICES_3;
  } else {
    // 5-LOD system (default): all LODs [0, 1, 2, 3, 4]
    thresholds = _LOD_THRESHOLDS_5;
    lodIndices = _LOD_INDICES_5;
  }

  // lodScale is the continuous FPS/VRAM control knob: scale the effective
  // on-screen size so the SAME picker chooses lower LODs sooner when the
  // controller wants cheaper frames (lodScale<1) or higher LODs when there's
  // headroom (lodScale>1).
  const effPx = screenPx * lodScale;

  // HYSTERESIS: thresholds are descending (more px = more detail). The ladder
  // index `i` counts how many thresholds effPx exceeds. Without a dead-band an
  // entity sitting right at a threshold (or whose effPx jitters as lodScale and
  // screenPx wobble) flips between adjacent LODs every frame — the continuous
  // vertcolor<->textured stutter, and the low<->high ping-pong. We bias the
  // comparison by the entity's CURRENT ladder index: to move UP a level effPx
  // must clear the threshold by +margin; to drop DOWN it must fall below by
  // -margin. Inside the band the current level is kept, so a still or slowly
  // moving entity stays put.
  const HYST = 0.18; // ±18% dead-band around each threshold
  // current ladder index (position in lodIndices) for the entity's curLod
  let curIdx = -1;
  if (curLod >= 0) { const p = lodIndices.indexOf(curLod); if (p >= 0) curIdx = p; }
  let i = 0;
  for (let t = 0; t < thresholds.length; t++) {
    const thr = thresholds[t];
    // crossing from below (gaining detail, going to ladder index t+1) needs +margin;
    // staying/falling uses -margin. Bias depends on whether we're currently above
    // this boundary already.
    const goingUpBoundary = curIdx <= t; // we'd be increasing past this threshold
    const eff = goingUpBoundary ? thr * (1 + HYST) : thr * (1 - HYST);
    if (effPx > eff) i++; else break;
  }
  if (ceilingIdx != null) i = Math.min(i, ceilingIdx);
  const clampedIdx = Math.min(i, lodIndices.length - 1);
  return use3LodSystem ? lodIndices[clampedIdx] : clampedIdx;
}
function _pickTexLod(lods, screenPx) {
  const target = Math.max(64, screenPx);
  let bestIdx = 0;
  for (let i = 0; i < lods.length; i++) {
    if (lods[i].width <= target * 2) bestIdx = i;
  }
  return bestIdx;
}

// Pre-compute optimal texture LODs for all textures at a given screen-space size.
// Called once when mesh transitions to a new LOD, replaces 1000+ per-frame _pickTexLod() calls.
function _precomputeAllTexLods(asset, screenPx) {
  const result = {};
  for (let tdIdx = 0; tdIdx < asset.texLodDescs.length; tdIdx++) {
    const desc = asset.texLodDescs[tdIdx];
    if (desc) result[tdIdx] = _pickTexLod(desc.lods, screenPx);
  }
  return result;
}

const _MAT_TEX_SLOTS = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap'];

// Which texLodDescs indices a material actually uses, by texture-name match.
// Drives per-mesh _applyTexLod scoping so a mesh never has a foreign
// descriptor's bitmap applied to it.
function _meshTexDescIdxs(mat, texLodDescs) {
  const idxs = [];
  if (!mat || !texLodDescs) return idxs;
  const names = new Set();
  for (const slot of _MAT_TEX_SLOTS) {
    const t = mat[slot];
    if (t && t.name) names.add(t.name);
  }
  if (!names.size) return idxs;
  for (let i = 0; i < texLodDescs.length; i++) {
    const d = texLodDescs[i];
    if (d && d.name && names.has(d.name)) idxs.push(i);
  }
  return idxs;
}

// Same texture-slot resolver as the inline demo.
function _findMaterialSlots(mat, texEntry) {
  if (!mat) return [];
  const out = new Set();
  for (const slot of _MAT_TEX_SLOTS) {
    const t = mat[slot];
    if (!t) continue;
    const tname = t.name || '';
    if (tname && texEntry.name && tname === texEntry.name) out.add(t);
  }
  if (out.size) return [...out];
  // Heuristic name-keyword match for assets whose texture names don't equal the
  // descriptor name exactly. NOTE: there is deliberately NO `mat.map` catch-all
  // here. The old `if (!out.size && mat.map) out.add(mat.map)` fallback stamped
  // ANY descriptor's bitmap into the base-colour map, which is what cross-wrote
  // the wrong texture onto multi-texture meshes on every LOD switch. Returning
  // empty (no write) is correct: a descriptor that matches nothing on this
  // material must not touch it.
  const nm = (texEntry.name || '').toLowerCase();
  if (nm.includes('normal') && mat.normalMap) out.add(mat.normalMap);
  if ((nm.includes('metallic') || nm.includes('roughness'))) {
    if (mat.roughnessMap) out.add(mat.roughnessMap);
    if (mat.metalnessMap) out.add(mat.metalnessMap);
  }
  return [...out];
}

// --- GPU VRAM Detection ---------------------------------------------------
function _detectAvailableVRAM() {
  // Try to detect available GPU VRAM using multiple heuristics.
  // Returns estimated VRAM in MB.

  // 1. Check navigator.deviceMemory if available (total system RAM in GB).
  if (typeof navigator !== 'undefined' && navigator.deviceMemory) {
    const systemRamGB = navigator.deviceMemory;
    // Heuristic: 50% of system RAM as GPU estimate (conservative).
    return Math.floor(systemRamGB * 512);
  }

  // 2. Try WebGL info queries if renderer available.
  // This is best-effort; most browsers don't expose exact VRAM.
  // Heuristic based on GPU vendor and market analysis:
  //   - Desktop NVIDIA/AMD: 2-4 GB typical
  //   - Integrated (Intel): 256-512 MB shared
  //   - Mobile: 256-512 MB
  try {
    if (typeof window !== 'undefined' && window.navigator) {
      const ua = window.navigator.userAgent.toLowerCase();
      if (ua.includes('mobile') || ua.includes('android') || ua.includes('iphone')) {
        return 512; // Mobile device
      }
      // Desktop: assume integrated GPU (conservative).
      return 2048; // 2 GB
    }
  } catch (e) {
    // Ignore errors, fall through to default.
  }

  // 3. Default conservative estimate.
  return 1024; // 1 GB
}

// --- ModelPool: the public facade -----------------------------------------
export class ModelPool extends Emitter {
  constructor(opts = {}) {
    super();
    this.scene = opts.scene;
    this.renderer = opts.renderer;
    this.camera = opts.camera;
    // Configure the shared KTX2Loader with the renderer so KHR_texture_basisu
    // textures transcode to GPU-compressed formats on load (detectSupport needs
    // the GL context). Done before any Asset/loader is created.
    if (this.renderer) _ensureKtx2Loader(this.renderer);
    this.targetFps = opts.targetFps ?? 50;
    // Material Grouping Optimization: Initialize global material pool for tier-based consolidation
    this._globalMaterialPool = new GlobalMaterialPool(this.renderer, opts);
    this._globalMaterialPool._useGlobalMaterialPool = opts.useGlobalMaterialPool !== false; // default enabled
    // BatchedMesh FAR tier: collapse all distinct far-asset draws into ~1.
    // Opt-in (default off) until witnessed at scale; enable via opts or
    // pool._useBatchedFarTier = true. When on, _getInstancedSlot returns a
    // shared-BatchedMesh adapter instead of per-asset InstancedBatch slots.
    this._useBatchedFarTier = opts.useBatchedFarTier === true;
    this._batchedFarTier = null;
    // Octahedral impostor FINAL LOD (opt-in, default off): below _impostorPx on
    // screen the whole entity collapses to ONE billboard sampling a per-asset
    // octahedral atlas (baked on-the-fly, in ONE InstancedMesh draw across all
    // assets). One rung past the BatchedFarTier. See octahedral-impostor-tier.js.
    this._useImpostorFinalLod = opts.useImpostorFinalLod === true;
    this._impostorTier = null;
    // GPU occlusion culling (opt-in, default off): entities that are
    // in-frustum but hidden behind other geometry (walls, dense neighbors)
    // skip their draw. Auto-picks a backend off the renderer: WebGL2-native
    // occlusion queries (ANY_SAMPLES_PASSED_CONSERVATIVE, occlusion-query-tier.js)
    // for a WebGLRenderer, or a WebGPU compute-based tier (webgpu-hiz-tier.js,
    // currently fail-open scaffolding pending real-hardware witnessing -- see
    // that file's header) for a WebGPURenderer with a real WebGPU backend.
    // Only worth enabling above minOcclusionCandidates entities/frame — the
    // per-entity query submission has its own cost.
    this._useOcclusionQuery = opts.useOcclusionQuery === true;
    this._occlusionTier = null;
    this._occlusionMinCandidates = opts.occlusionMinCandidates ?? 64;
    this._impostorPx = opts.impostorPx ?? 14;       // enter impostor below this px
    this._impostorGrid = opts.impostorGrid ?? 8;
    this._impostorCellPx = opts.impostorCellPx ?? 64;
    // Cells baked per frame. Each cell re-renders the WHOLE model, so per-frame
    // bake cost is ~budget * model-draw-cost; keep it low (4) so even a heavy
    // multi-mesh map only adds a few renders/frame. A 64-cell atlas then takes
    // ~16 frames (~0.27s) to fully bake, which is fine for a far LOD.
    this._impostorCellBudget = opts.impostorCellBudget ?? 4;
    this._impostorBakeQueue = new Map();                      // asset.url -> entity
    this._impostorBlend = opts.impostorBlend === true;        // bilinear view cross-fade
    this._impostorPadding = opts.impostorPadding;             // cell gutter; undefined -> tier default 1.05
    // EZ impostor path (opt-in, default off): localized @three.ez/octahedron-
    // imposter -> LIT impostors (albedo + baked normal/depth) via a 1024 MRT
    // atlas per asset, capped by impostorMaxAssets. Per-asset InstancedMesh
    // (N draws for N distinct assets) vs the array-texture tier's 1-draw, traded
    // for scene-lit quality + ~1M-tri source support. See octahedral-impostor-ez-tier.js.
    this._useImpostorEz = opts.useImpostorEz === true;
    this._impostorTextureSize = opts.impostorTextureSize ?? 1024;
    this._impostorMaxAssets = opts.impostorMaxAssets ?? 64;
    this._impostorHemiOcta = opts.impostorHemiOcta === true;
    // Eager-allocate the impostor tier at construction so its array-texture VRAM
    // allocation lands at startup, NOT on the first swap (where it was a one-shot
    // ~120ms+ frame hitch). No-op when impostors are disabled.
    if (this._useImpostorFinalLod) this._getImpostorTier();
    this._impostorActiveCount = 0;                  // stat: entities on impostors
    // GPU instance-transform texture: OPT-IN (default off). It was FPS-neutral
    // (static transforms upload once either way) and at 500 distinct it placed
    // instances off-screen/degenerate — the screen went ~empty (1.2% coverage)
    // while stats still reported ~497 "visible". The proven instanceMatrix
    // attribute path (default) renders correctly (64% coverage). Re-enable only
    // with opts.enableGpuInstanceTex once the at-scale texel correctness is fixed.
    this._enableGpuInstanceTex = opts.enableGpuInstanceTex === true;
    // Continuous FPS-control knob: multiplier on per-entity screen size in the
    // LOD picker. <1 = models drop to lower LODs sooner (cheaper); >1 = detail
    // extends farther. Driven by the closed-loop controller in update().
    this._lodDistanceScale = 1;
    // Frame-budgeted LOD warm loader: LODs the picker wants but that aren't yet
    // GPU-resident are registered here and processed PIECEMEAL, only when FPS
    // has headroom, so fetch+decode+GPU-upload never lands on a switch frame.
    // Network stays lazy (fetch on demand); GPU upload is eager once decoded.
    this._lodWarmQueue = new Map();   // key -> { asset, meshDescIdx, lodIdx, dist }
    this._lodWarmInFlight = 0;
    this._lodWarmMaxInFlight = opts.lodWarmMaxInFlight ?? 3; // concurrent decodes
    this._lodWarmPerFrame = opts.lodWarmPerFrame ?? 2;        // starts per frame (headroom-gated)
    this._gpuWarmPending = [];        // decoded geos awaiting GPU upload, 1/frame
    // Asset Streaming: Deferred load queue + unload manager
    this._deferredLoadQueue = new DeferredLoadQueue(opts.maxConcurrentDefers ?? 2);
    this._lodUnloadManager = new LodUnloadManager(opts.vramBudgetMB ?? 200);
    this._enableDeferredStreaming = opts.enableDeferredStreaming !== false; // Re-enabled with proper timeout handling
    // Detect available GPU VRAM and estimate safe budget (60-70% of available).
    this._estimatedVramMB = _detectAvailableVRAM();
    const safeByteBudget = Math.floor((this._estimatedVramMB * 0.65) * 1024 * 1024);
    this.byteBudget = opts.byteBudget ?? safeByteBudget;
    this._budgetAdjustmentCooldown = 0; // prevent too-frequent budget changes
    // VRAM-aware dynamic pool sizing: monitor actual GPU memory ratio
    this._vramRatioMonitor = {
      currentRatio: 0,           // current usage ratio (0-1)
      peakRatio: 0,              // peak ratio this session
      lastAdjustmentRatio: 0,    // last ratio at which we adjusted
      adjustmentCooldown: 0,     // frames since last dynamic adjustment
    };
    // Wide gap between critical (lower ceiling) and safe (relax ceiling) so the
    // VRAM monitor does not cycle critical<->safe and toggle LOD ceilings, which
    // makes models pop in/out. Old 0.70/0.40 with a 30-frame cooldown oscillated;
    // 0.85/0.45 with longer cooldowns (set at the adjust sites) is stable.
    this._vramThresholdWarning = 0.78;  // warn at 78%
    this._vramThresholdCritical = 0.85; // adjust LOD ceiling at 85%
    this._vramThresholdSafe = 0.45;     // allow relaxation at 45%
    this.maxConcurrentFetches = opts.maxConcurrentFetches ?? 6;
    this.animationThrottleDistance = opts.animationThrottleDistance ?? 15; // More aggressive animation throttling for better FPS
    this._assets = new Map(); // url -> Asset
    this._entities = new Set();
    // Position-update / lerp system. Only entities with an ACTIVE target live in
    // this map, so the per-frame cost is O(moving entities), not O(all). Idle
    // entities are never touched. Each record interpolates root.position from
    // (x0,y0,z0) -> (x1,y1,z1) over [start, start+dur]; on completion the entity
    // is removed from the map and its matrix is left resting at the target.
    this._movers = new Map(); // entity -> { x0,y0,z0, x1,y1,z1, start, dur }
    // Far-tier GPU-lerp records (entity -> same shape). Written only on setTarget
    // (O(1), never per-frame); used solely to sample current position for
    // continuous mid-flight retargets. The actual interpolation is on the GPU.
    this._farLerpState = new Map();
    this._nextEntityId = 0;
    this._totalBytes = 0;
    this._byteLog = new Map(); // assetUrl -> { url -> bytes }
    this._loadQueue = new Map(); // dedupe key -> Promise
    this._inFlight = 0;
    this._pending = []; // queued tasks { key, run, resolve, reject }
    this._fpsEma = 60;
    this._lastTick = performance.now();
    this._fpsGoodFrames = 0; // counter for ceiling relaxation (requires 5 consecutive frames above target)
    this._budgetLowFrames = 0; // counter for budget increase (requires 10 frames below 40% utilization)
    this._currentCeilingLod = 4; // Start at LOD 4 for maximum GPU instancing (all entities FAR tier)
    this._frustum = new THREE.Frustum();
    this._tmpMatrix = new THREE.Matrix4();
    // Dynamic frustum caching: track scene staticness and adjust check interval
    this._frustumCheckInterval = 5; // Default 5-frame interval for static entities
    this._dynamicFrustumCheckInterval = 5; // Will be updated based on scene staticness
    this._lastFrameMovingCount = 0; // Number of entities that moved last frame
    // Distance-coordinated bucket system: per-category per-frame budgets in milliseconds
    this._heroBudgetMs = 2.0;   // HERO tier: fully animated entities, ~2ms budget
    this._midBudgetMs = 4.0;    // MID tier: mixed quality, ~4ms budget
    this._heroDist = 20;        // Distance threshold for HERO tier (meters)
    this._midDist = 60;         // Distance threshold for MID tier (meters)
    this._heroFrameTimeMs = 0;  // Accumulated per-frame time for HERO tier
    this._midFrameTimeMs = 0;   // Accumulated per-frame time for MID tier
    this._entityDistances = []; // Cache: [{ entity, distance, screenPx }] sorted by distance
    // Stats snapshot — refreshed each tick, exposed via getStats().
    this._stats = { fps: 0, entities: 0, drawCalls: 0, ceilingLod: null, bytes: 0, assets: 0, inFlight: 0, hero: 0, mid: 0, far: 0, heroBudgetMs: 0, midBudgetMs: 0 };
    // Shared InstancedMesh slots: key `${assetUrl}|${meshDescIdx}|${lodIdx}` -> InstancedSlot.
    this._instancedSlots = new Map();
    // Tier thresholds (in screen pixels of the entity's bounding sphere).
    // The three tiers are entirely a function of which LOD the picker
    // selects, NOT a separate routing layer:
    //   HERO: top-of-ladder textured LODs (per-entity SkinnedMesh draws).
    //   MID:  middle textured / vertcolor LODs (still per-entity draws, but
    //         the geometry is decimated and the material is cheaper).
    //   FAR:  unskinned LOD (routes through per-asset InstancedMesh; real 3D
    //         geometry preserved — works for terrain chunks, props, anything).
    // We track these in the HUD only for visibility; routing happens via the
    // standard LOD picker.
    this.heroPx = opts.heroPx ?? 200;
    this.midPx = opts.midPx ?? 120; // Aggressive default: push more to FAR tier for speed
    this.heroCap = opts.heroCap ?? 3;  // Maximum 3 HERO entities to prioritize GPU instancing
    // Feature toggles for Phase 5 validation
    this._enableFrustumCulling = true;
    this._enableTextureLod = true;
    this._enableAnimThrottle = true;
    // 3-LOD Simplification: use only LODs [0, 2, 4], skip intermediate LODs 1 and 3
    // This saves ~40% VRAM and reduces memory allocation churn (default enabled)
    // 5-LOD ladder by default now. The 3-LOD system used only LODs [0,2,4],
    // which made entities jump straight between the lowest (unskinned) and
    // highest (textured) tiers with nothing in between — the "swaps straight
    // from lowest to highest instead of a good range" symptom. The full
    // [0,1,2,3,4] ladder gives a smooth gradient; epoch-gated picking + picker
    // hysteresis keep it from churning. Opt back into 3-LOD via opts.use3LodSystem.
    this._use3LodSystem = opts.use3LodSystem === true; // default false (5-LOD)
    // Worker pool for sibling-LOD fetch + decode. Defaults to 4 workers
    // (more = more concurrent decodes; each holds one three.js instance so
    // memory grows linearly). Set to 0 to disable and fall back to
    // main-thread decode.
    // Scale decode-worker parallelism to the machine (leave one core for the
    // main thread), clamped to a sane 2..8. Explicit opts.workerCount wins.
    this._workerCount = opts.workerCount ?? Math.max(2, Math.min(8, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4) - 1));
    this._workers = [];
    this._workerRR = 0; // round-robin cursor
    this._workerPending = new Map(); // id -> {resolve, reject}
    this._workerNextId = 0;
    if (this._workerCount > 0 && typeof Worker !== 'undefined') {
      try {
        for (let i = 0; i < this._workerCount; i++) {
          const workerUrl = new URL('./lod-worker.js', import.meta.url);
          const w = new Worker(workerUrl, { type: 'module' });
          w.addEventListener('message', (ev) => {
            const m = ev.data;
            if (m && m.id === 0 && m.ready) {
              if (!m.ok) console.error('[pool] worker init failed:', m.error);
              else console.log('[pool] worker ready');
              return;
            }
            this._onWorkerMessage(m);
          });
          w.addEventListener('error', (e) => {
            const detail = {
              message: e.message || '(no message)',
              filename: e.filename || '(no filename)',
              lineno: e.lineno,
              colno: e.colno,
              error: e.error ? (e.error.stack || String(e.error)) : '(no error obj)',
              workerUrl: String(workerUrl),
            };
            console.error('[pool] worker error', JSON.stringify(detail));
          });
          w.addEventListener('messageerror', (e) => console.error('[pool] worker messageerror', String(e)));
          this._workers.push(w);
        }
      } catch (e) {
        console.warn('[pool] worker init failed, falling back to main-thread decode', e);
        this._workers = [];
      }
    }

    // Phase 3 Quick-Wins Optimizations
    // QW1: Vertex Attribute Compression (pack vec4 → vec3)
    this._vertexCompressionOptimizer = opts.enableVertexCompression !== false
      ? new VertexCompressionOptimizer()
      : null;

    // QW4: Instance Reuse Across Assets (hash LOD geometry, reuse InstancedSlots)
    this._enableInstanceReuse = opts.enableInstanceReuse !== false; // default enabled
    this._lodGeometryHash = new Map(); // geometry content hash → LOD info

    // QW5: Attribute Deinterleaving (separate position/normal/uv buffers)
    this._enableDeinterleaving = opts.enableDeinterleaving !== false; // default enabled
  }

  // Phase 5: Public getter/setter for interactive UI control
  get ceilingLod() {
    return this._currentCeilingLod;
  }
  set ceilingLod(val) {
    this._currentCeilingLod = val === 5 ? null : (val != null ? val : null);
  }

  // Public getter/setter for frustum check interval override (for UI slider)
  get frustumCheckInterval() {
    return this._frustumCheckInterval;
  }
  set frustumCheckInterval(val) {
    // Allow manual override: if set to 0, use automatic dynamic calculation
    // Otherwise, force fixed interval for testing/tuning
    this._frustumCheckInterval = val;
    if (val === 0) {
      // Re-enable dynamic calculation
      this._dynamicFrustumCheckInterval = 5;
    }
  }

  _onWorkerMessage(msg) {
    const pend = this._workerPending.get(msg.id);
    if (!pend) return;
    this._workerPending.delete(msg.id);
    if (msg.ok) pend.resolve(msg.payload);
    else pend.reject(new Error(msg.error || 'worker decode failed'));
  }

  // Fetch + decode a sibling LOD GLB in a worker; returns a Promise<payload>
  // where payload is { attrs, index, boundingSphere, boundingBox, bytes }.
  _workerFetchLod(url, decodeAABB, sloppyCap = 0) {
    if (!this._workers.length) return null;
    const id = ++this._workerNextId;
    const w = this._workers[this._workerRR];
    this._workerRR = (this._workerRR + 1) % this._workers.length;
    return new Promise((resolve, reject) => {
      this._workerPending.set(id, { resolve, reject });
      w.postMessage({ id, url, decodeAABB, sloppyCap });
    });
  }

  // Rebuild a BufferGeometry on the main thread from a worker payload.
  // No heavy work here — typed arrays were transferred, so this is just
  // attribute wiring.
  static _buildGeometryFromPayload(payload) {
    const geo = new THREE.BufferGeometry();
    for (const k of Object.keys(payload.attrs)) {
      const a = payload.attrs[k];
      // The worker denormalizes every attribute to 0..1 Float32 via getX/getY/getZ,
      // but ships the SOURCE `normalized` flag. If we pass normalized:true with a
      // Float32Array, THREE re-normalizes already-0..1 values and corrupts them —
      // this is why vertex colors rendered white/black. A Float32Array is never
      // a normalized integer buffer, so force normalized:false for float arrays.
      const isFloat = a.array instanceof Float32Array;
      const normalized = isFloat ? false : !!a.normalized;
      geo.setAttribute(k, new THREE.BufferAttribute(a.array, a.itemSize, normalized));
    }
    if (payload.index) {
      geo.setIndex(new THREE.BufferAttribute(payload.index, 1));
    }
    if (payload.boundingSphere) {
      geo.boundingSphere = new THREE.Sphere(
        new THREE.Vector3().fromArray(payload.boundingSphere.center),
        payload.boundingSphere.radius
      );
    }
    if (payload.boundingBox) {
      geo.boundingBox = new THREE.Box3(
        new THREE.Vector3().fromArray(payload.boundingBox.min),
        new THREE.Vector3().fromArray(payload.boundingBox.max)
      );
    }
    return geo;
  }

  // QW1: Apply vertex attribute compression (vec4 → vec3) if enabled
  _compressGeometryAttributes(geometry) {
    if (!this._vertexCompressionOptimizer) return geometry;
    return this._vertexCompressionOptimizer.compressGeometry(geometry);
  }

  // QW4: Generate a hash of LOD geometry for reuse detection.
  // Hash is based on vertex count, index count, and bounding sphere.
  // Geometries with identical hashes can share InstancedSlots.
  _hashLodGeometry(geometry) {
    if (!geometry) return null;
    const posAttr = geometry.getAttribute('position');
    const indexAttr = geometry.getIndex();
    const bounds = geometry.boundingSphere;
    if (!posAttr) return null;

    // Simple hash: vertex count + index count + radius
    // In production, could use a cryptographic hash of actual vertex data
    const vertCount = posAttr.count;
    const indexCount = indexAttr ? indexAttr.count : 0;
    const radius = bounds ? bounds.radius.toFixed(2) : '0';
    return `${vertCount}:${indexCount}:${radius}`;
  }

  // QW4: Check if a geometry hash already has a reusable InstancedSlot.
  // If yes, returns the slot; otherwise returns null.
  _findReusableSlotByGeometryHash(geometry) {
    if (!this._enableInstanceReuse) return null;
    const hash = this._hashLodGeometry(geometry);
    if (!hash) return null;
    const info = this._lodGeometryHash.get(hash);
    return info ? info.slot : null;
  }

  // QW4: Register a geometry hash → slot mapping for future reuse.
  _registerGeometryHashSlot(geometry, slot) {
    if (!this._enableInstanceReuse) return;
    const hash = this._hashLodGeometry(geometry);
    if (!hash) return;
    const info = { slot, geometry, refCount: 1 };
    this._lodGeometryHash.set(hash, info);
  }

  // QW5: Deinterleave vertex buffer layout.
  // Converts interleaved (pos, norm, uv, pos, norm, uv, ...)
  // to separate (pos..., norm..., uv...)
  // This improves L1/L2 cache hit rate for vertex shader.
  _deinterleaveGeometryAttributes(geometry) {
    if (!this._enableDeinterleaving) return geometry;

    const posAttr = geometry.getAttribute('position');
    const normAttr = geometry.getAttribute('normal');
    const uvAttr = geometry.getAttribute('uv');

    if (!posAttr) return geometry; // No position = can't deinterleave

    const vertCount = posAttr.count;

    // If already deinterleaved (separate underlying ArrayBuffers), skip.
    // NOTE: THREE.BufferAttribute has no `.buffer` property (only
    // `.array.buffer`, the underlying ArrayBuffer of the typed array; for an
    // InterleavedBufferAttribute, `.array` is a getter that returns
    // `.data.array`, so this same check also works for the interleaved case).
    // The old `posAttr.buffer !== normAttr?.buffer` compared two `undefined`s
    // and was always false, so this guard never short-circuited: every LOD
    // load paid a full O(vertexCount) copy + a geometry.clone() even when the
    // geometry was already deinterleaved.
    if (posAttr.array.buffer !== normAttr?.array?.buffer || posAttr.array.buffer !== uvAttr?.array?.buffer) {
      return geometry; // Already deinterleaved or not interleaved
    }

    // Allocate separate buffers
    const newPosArray = new Float32Array(vertCount * 3);
    const newNormArray = normAttr ? new Float32Array(vertCount * 3) : null;
    const newUvArray = uvAttr ? new Float32Array(vertCount * 2) : null;

    // Copy data into deinterleaved layout
    for (let i = 0; i < vertCount; i++) {
      newPosArray[i * 3 + 0] = posAttr.getX(i);
      newPosArray[i * 3 + 1] = posAttr.getY(i);
      newPosArray[i * 3 + 2] = posAttr.getZ(i);

      if (newNormArray && normAttr) {
        newNormArray[i * 3 + 0] = normAttr.getX(i);
        newNormArray[i * 3 + 1] = normAttr.getY(i);
        newNormArray[i * 3 + 2] = normAttr.getZ(i);
      }

      if (newUvArray && uvAttr) {
        newUvArray[i * 2 + 0] = uvAttr.getX(i);
        newUvArray[i * 2 + 1] = uvAttr.getY(i);
      }
    }

    // Create new deinterleaved attributes
    const newGeo = geometry.clone();
    newGeo.setAttribute('position', new THREE.BufferAttribute(newPosArray, 3));
    if (newNormArray) newGeo.setAttribute('normal', new THREE.BufferAttribute(newNormArray, 3));
    if (newUvArray) newGeo.setAttribute('uv', new THREE.BufferAttribute(newUvArray, 2));

    return newGeo;
  }

  // Register a wanted-but-not-resident LOD for the frame-budgeted warm loader.
  // Cheap + idempotent: just records the want (nearest distance wins priority).
  _enqueueLodWarm(asset, meshDescIdx, lodIdx, dist) {
    const desc = asset.meshLodDescs[meshDescIdx];
    if (!desc) return;
    const t = desc.lods[lodIdx];
    if (!t || t.inline) return; // inline LODs are always resident
    const key = `${asset.url}#${desc.meshIndex}:${desc.primIndex}:${lodIdx}`;
    if (asset.geoCache.has(`${desc.meshIndex}:${desc.primIndex}:${lodIdx}`)) return; // already resident
    const ex = this._lodWarmQueue.get(key);
    if (ex) { if (dist < ex.dist) ex.dist = dist; return; }
    this._lodWarmQueue.set(key, { asset, meshDescIdx, lodIdx, dist });
  }

  // ---- Position-update API: GPU-eager, CPU-O(moving) ---------------------
  // Move an entity toward a target over durationMs. The CPU only records the
  // target here; the per-frame interpolation runs in _drainMovers() over just
  // the active set. A retarget mid-flight is continuous: the current
  // interpolated position becomes the new start. durationMs<=0 snaps instantly.
  setTarget(entity, x, y, z, durationMs = 300, nowMs = (typeof performance !== 'undefined' ? performance.now() : Date.now())) {
    if (!entity || entity._disposed) return;
    // Determine the start position = current rendered position (continuous
    // retarget). If a mover is already in flight, sample it at `now`; else use
    // the entity's resting root.position.
    let sx, sy, sz;
    const cur = this._movers.get(entity);
    const farCur = this._farLerpState && this._farLerpState.get(entity);
    if (cur) {
      const t = cur.dur > 0 ? Math.min(1, Math.max(0, (nowMs - cur.start) / cur.dur)) : 1;
      sx = cur.x0 + (cur.x1 - cur.x0) * t;
      sy = cur.y0 + (cur.y1 - cur.y0) * t;
      sz = cur.z0 + (cur.z1 - cur.z0) * t;
    } else if (farCur) {
      // Continuous retarget for a far entity mid-GPU-lerp: sample its current
      // interpolated position (mirror of the shader math) as the new start.
      const t = farCur.dur > 0 ? Math.min(1, Math.max(0, (nowMs - farCur.start) / farCur.dur)) : 1;
      sx = farCur.x0 + (farCur.x1 - farCur.x0) * t;
      sy = farCur.y0 + (farCur.y1 - farCur.y0) * t;
      sz = farCur.z0 + (farCur.z1 - farCur.z0) * t;
    } else {
      const p = entity.root.position;
      sx = p.x; sy = p.y; sz = p.z;
    }
    if (durationMs <= 0) {
      // Instant: write position now, no mover record needed.
      this._movers.delete(entity);
      if (this._farLerpState) this._farLerpState.delete(entity);
      this._clearFarLerp(entity);
      this._applyEntityPosition(entity, x, y, z);
      return;
    }
    // GPU-LERP FAST PATH: if every tracked mesh of this entity lives on the
    // BatchedMesh far tier, push the interpolation onto the GPU (write 2 texels)
    // and DO NOT register a CPU mover — the vertex shader interpolates each
    // frame, so the CPU writes nothing per-frame for this entity (only this
    // sparse texel write + the once-per-frame uNow uniform). Falls through to the
    // CPU mover for hero/mid tiers the shader lerp can't reach.
    if (this._setFarLerp(entity, sx, sy, sz, x, y, z, nowMs / 1000, durationMs / 1000)) {
      this._movers.delete(entity); // ensure no stale CPU mover double-drives it
      // Record the lerp CPU-side (O(1), only on retarget — NOT per frame) so a
      // mid-flight retarget can sample the true current position for continuity.
      this._farLerpState.set(entity, { x0: sx, y0: sy, z0: sz, x1: x, y1: y, z1: z, start: nowMs, dur: durationMs });
      // Set root.position to the current START (not the target): distance/LOD
      // read where the entity actually is now; it will settle toward the target
      // as the GPU lerps. (One write, not per-frame.)
      entity.root.position.set(sx, sy, sz);
      return;
    }
    this._movers.set(entity, { x0: sx, y0: sy, z0: sz, x1: x, y1: y, z1: z, start: nowMs, dur: durationMs });
  }

  // Try to drive an entity's interpolation entirely on the GPU via the far tier.
  // Returns true only if the entity is fully resident on the BatchedFarTier (all
  // tracked meshes batched-far with a valid instance id), so the CPU can skip it.
  _setFarLerp(entity, x0, y0, z0, x1, y1, z1, startSec, durSec) {
    const tier = this._batchedFarTier;
    if (!tier) return false;
    if (!entity.trackedMeshes || entity.trackedMeshes.length === 0) return false;
    const id = tier.instanceIdFor(entity);
    if (id < 0) return false;
    // Every tracked mesh must be on the far tier (else part of the entity would
    // not interpolate). In practice the far tier carries the whole entity.
    for (const tm of entity.trackedMeshes) {
      if (!tm._instancedSlot || !tm._instancedSlot._batchedFar) return false;
    }
    tier.setLerpTarget(id, x0, y0, z0, x1, y1, z1, startSec, durSec);
    return true;
  }

  _clearFarLerp(entity) {
    if (this._farLerpState) this._farLerpState.delete(entity);
    const tier = this._batchedFarTier;
    if (!tier) return;
    const id = tier.instanceIdFor(entity);
    if (id >= 0) tier.clearLerp(id);
  }

  // Write an absolute position onto an entity and push it to whatever slot(s)
  // its tracked meshes currently occupy (far BatchedMesh, mid InstancedMesh, or
  // hero). Backend-agnostic: both tiers route through setMatrixForSlot.
  _applyEntityPosition(entity, x, y, z) {
    entity.root.position.set(x, y, z);
    // Static entities have matrixAutoUpdate off; refresh the world matrix once.
    entity.root.updateMatrix();
    entity.root.updateMatrixWorld(true);
    for (const tm of entity.trackedMeshes) {
      if (tm._instancedSlot && tm._instancedSlotIdx != null && tm._instancedSlotIdx >= 0) {
        tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, entity._slotWorldMatrix(tm));
      }
    }
  }

  // Set an entity's ROTATION (quaternion [x,y,z,w]) and refresh its rendered transform
  // on every tier. setTarget()/the far-tier GPU lerp interpolate POSITION only, so without
  // this a networked physics/kinematic body's rotation never reaches the pool render (the
  // consumer sees a body that translates but never turns). Writing root.quaternion here and
  // re-deriving every tracked mesh's slot matrix from _slotWorldMatrix(tm) (which reads the
  // full root TRS) carries rotation to the hero/mid Object3D tier AND the batched/instanced
  // far tier. Rotation is snapped (authoritative server quat at the snapshot rate); position
  // keeps its own smooth lerp, so a small rotation step per snapshot reads as continuous.
  setRotation(entity, qx, qy, qz, qw) {
    if (!entity || entity._disposed || !entity.root) return;
    entity.root.quaternion.set(qx, qy, qz, qw);
    entity.root.updateMatrix();
    entity.root.updateMatrixWorld(true);
    for (const tm of entity.trackedMeshes) {
      if (tm._instancedSlot && tm._instancedSlotIdx != null && tm._instancedSlotIdx >= 0) {
        tm._instancedSlot.setMatrixForSlot(tm._instancedSlotIdx, entity._slotWorldMatrix(tm));
      }
    }
  }

  // Per-frame: interpolate only entities with an active target. O(active set).
  // Completed movers are removed and left resting at their target. Called once
  // per frame from update().
  _drainMovers(nowMs) {
    if (this._movers.size === 0) return;
    for (const [entity, m] of this._movers) {
      if (entity._disposed) { this._movers.delete(entity); continue; }
      const t = m.dur > 0 ? Math.min(1, Math.max(0, (nowMs - m.start) / m.dur)) : 1;
      const x = m.x0 + (m.x1 - m.x0) * t;
      const y = m.y0 + (m.y1 - m.y0) * t;
      const z = m.z0 + (m.z1 - m.z0) * t;
      this._applyEntityPosition(entity, x, y, z);
      if (t >= 1) this._movers.delete(entity); // settled: stop touching it
    }
  }

  // Process the warm queue PIECEMEAL: only when FPS has headroom, start at most
  // _lodWarmPerFrame decodes/frame and cap concurrent in-flight. Each item
  // fetches (lazy) + decodes (worker) via ensureMeshLod, then GPU-warms the
  // resulting geometry so the eventual LOD switch pays no first-use upload.
  // Nearest-first. Called once per frame from update().
  _drainLodWarm() {
    if (this._lodWarmQueue.size === 0) return;
    // Adaptive budget: more starts when FPS has headroom, but NEVER zero —
    // gating warming off entirely below target deadlocks (entities that need a
    // cheaper far LOD to recover FPS can never get it, so FPS stays low forever
    // and the queue never drains). Always allow at least 1 decode/frame so the
    // scene can progress toward its resting LODs; ramp up with headroom.
    const target = this.targetFps || 60;
    let starts;
    if (this._fpsEma >= target + 5) starts = this._lodWarmPerFrame * 2;
    else if (this._fpsEma >= target - 8) starts = this._lodWarmPerFrame;
    else starts = 1; // struggling — minimum forward progress, no zero-deadlock
    // Pick nearest-first without sorting the whole map every frame: a cheap
    // single linear min-scan per start (queue is small in practice).
    while (starts-- > 0 && this._lodWarmInFlight < this._lodWarmMaxInFlight && this._lodWarmQueue.size) {
      let bestKey = null, bestDist = Infinity;
      for (const [k, v] of this._lodWarmQueue) { if (v.dist < bestDist) { bestDist = v.dist; bestKey = k; } }
      if (bestKey == null) break;
      const item = this._lodWarmQueue.get(bestKey);
      this._lodWarmQueue.delete(bestKey);
      this._lodWarmInFlight++;
      Promise.resolve(item.asset.ensureMeshLod(item.meshDescIdx, item.lodIdx))
        .then((geo) => { if (geo && !geo.__gpuWarmed) this._gpuWarmPending.push(geo); })
        .catch(() => {})
        .finally(() => { this._lodWarmInFlight--; });
    }
  }

  // Synchronous GPU warm (a renderer.render() into a 1x1 target) is the one
  // spiky part of warming: several decode promises can resolve in the same
  // frame and, done inline, fire N back-to-back renders → a frame-time burst
  // that shows up as the min-FPS dip during streaming. Decouple decode (kept
  // parallel) from upload (drained here at most ONE per frame) so the cost is
  // spread across frames instead of bunched. Called once per frame in update().
  _drainGpuWarm() {
    if (!this._gpuWarmPending.length) return;
    // Adaptive per-frame upload budget: spread uploads to avoid the burst that
    // caused the min-FPS dip, but NEVER throttle so hard that a large scene
    // can't reach its resting LODs. With headroom, drain aggressively (the
    // backlog clears fast during initial streaming); near/below target, drip.
    const target = this.targetFps || 60;
    let budget;
    if (this._fpsEma >= target + 5) budget = 8;
    else if (this._fpsEma >= target - 8) budget = 3;
    else budget = 1; // struggling — one upload/frame, still forward progress
    while (budget-- > 0 && this._gpuWarmPending.length) {
      const geo = this._gpuWarmPending.shift();
      if (geo && !geo.__gpuWarmed) this._gpuWarmGeometry(geo);
    }
  }

  // Eagerly upload a geometry's buffers to the GPU so a later draw with it does
  // not stall on first use. THREE uploads attributes lazily at first render;
  // renderer.initGeometry (r0.16x+ via WebGLAttributes) isn't public, so we use
  // the documented path: renderer.compile won't upload geometry, but rendering
  // it once into the current target does. To avoid a visible flash we draw it
  // through an off-screen 1x1 scratch with a tiny ortho cam. Cheap (1 tri batch
  // of already-decimated geo) and one-shot per geometry.
  _gpuWarmGeometry(geo) {
    if (!geo || geo.__gpuWarmed) return;
    try {
      if (!this._warmScene) {
        this._warmScene = new THREE.Scene();
        this._warmCam = new THREE.Camera();
        this._warmMat = new THREE.MeshBasicMaterial();
        this._warmMesh = new THREE.Mesh(undefined, this._warmMat);
        this._warmMesh.frustumCulled = false;
        this._warmScene.add(this._warmMesh);
        this._warmTarget = new THREE.WebGLRenderTarget(1, 1);
      }
      const prevTarget = this.renderer.getRenderTarget();
      this._warmMesh.geometry = geo;
      this.renderer.setRenderTarget(this._warmTarget);
      this.renderer.render(this._warmScene, this._warmCam); // forces buffer upload
      this.renderer.setRenderTarget(prevTarget);
      this._warmMesh.geometry = undefined;
      geo.__gpuWarmed = true;
    } catch (e) { /* warming is best-effort; a cold first-use is the fallback */ }
  }

  // Get-or-create an InstancedSlot for an (asset, meshDescIdx, lod) tuple.
  // Returns null if the LOD isn't suitable for instancing (currently only
  // 'unskinned' LODs qualify — they have no per-instance bone state).
  //
  // MATERIAL GROUPING OPTIMIZATION: Uses global material pool for FAR tier
  // instead of creating per-LOD materials. This reduces material count from
  // 8-12 to 3 (one per tier), reducing shader programs and state changes.
  // Register an asset for on-the-fly impostor baking, baked from `entity`'s
  // resident geometry. Drained with a per-frame CELL budget in update().
  _queueImpostorBake(asset, entity) {
    if (!this._impostorBakeQueue) this._impostorBakeQueue = new Map();
    if (this._impostorTier && this._impostorTier.hasAsset(asset)) return;
    if (!this._impostorBakeQueue.has(asset.url)) this._impostorBakeQueue.set(asset.url, entity);
  }

  // Drain the impostor bake queue using a per-frame CELL budget: render at most
  // _impostorCellBudget octahedral cells total this frame, spread across queued
  // assets via the tier's RESUMABLE bakeChunk. This is the swap-stall fix — a
  // baking frame costs a few small cell-renders instead of a whole grid*grid
  // atlas (which was tens-to-160+ms for heavy assets). Runs regardless of
  // per-entity fast-skip, so a still camera still converges to full coverage.
  _drainImpostorBakes() {
    const q = this._impostorBakeQueue;
    const tier = this._impostorTier;
    if (!q || q.size === 0 || !tier) return;
    // Headroom-gated cell budget (mirrors the frame-budgeted LOD warm loader):
    // when FPS sits above target there is slack to converge impostor coverage
    // faster (more distant models reach the 1-draw impostor sooner); when below
    // target, back off toward 0 so baking never deepens a frame-time deficit.
    // The base budget stays small so even the boosted path is a few cell-renders.
    const fps = this._fpsEma || 60;
    const target = this.targetFps || 50;
    let budget = this._impostorCellBudget;
    if (fps > target + 10) budget = this._impostorCellBudget * 2;      // slack: converge faster
    else if (fps < target) budget = Math.max(1, this._impostorCellBudget >> 1); // deficit: back off
    for (const [url, entity] of q) {
      if (budget <= 0) break;
      if (!entity || entity._disposed) { q.delete(url); continue; }
      if (tier.hasAsset(entity.asset)) { q.delete(url); continue; }
      budget -= tier.bakeChunk(entity.asset, entity.root, budget);
      if (tier.hasAsset(entity.asset)) q.delete(url);
    }
  }

  // Lazily build the shared octahedral impostor tier and attach its single
  // InstancedMesh to the scene. Null if impostors are disabled or unsupported.
  // App-facing: call ONCE per frame AFTER renderer.render(scene, camera) so
  // occlusion queries read against this frame's real depth buffer. Results
  // are read back and applied starting NEXT frame's update() (see the
  // occlusion gate at the top of the entity loop above) — one frame of
  // latency, standard for GPU occlusion query culling, avoids a sync stall.
  // No-op when useOcclusionQuery is off or the candidate count is below
  // occlusionMinCandidates (query submission overhead isn't worth it for a
  // handful of entities that frustum culling already handles cheaply).
  runOcclusionQueries() {
    if (!this._useOcclusionQuery) return;
    const tier = this._getOcclusionTier();
    if (!tier) return;
    const candidates = this._occlusionCandidates;
    if (!candidates || candidates.length < this._occlusionMinCandidates) return;
    tier.runQueries(this.camera, candidates);
  }

  _getOcclusionTier() {
    if (!this._useOcclusionQuery) return null;
    // A WebGPURenderer path is loaded via dynamic import() ONLY when the
    // renderer actually looks like one (cheap sync flag check, no module
    // load) -- so a WebGL2-only page (the overwhelming majority of this
    // repo's demos) never pulls in webgpu-hiz-tier.js or its 'three/tsl'
    // dependency at all. The import is async; until it resolves,
    // _getOcclusionTier() returns null (fail-open: no occlusion culling
    // that frame, exactly like "unsupported") rather than blocking the
    // synchronous entity-loop caller.
    if (this.renderer && this.renderer.isWebGPURenderer && !this._occlusionTier && !this._webgpuTierLoading) {
      this._webgpuTierLoading = true;
      import('./webgpu-hiz-tier.js').then(({ WebGpuHizTier }) => {
        this._webgpuTierLoading = false;
        if (!this._useOcclusionQuery || this._occlusionTier) return; // opted out or a tier already assigned while loading
        const gpuTier = new WebGpuHizTier(this.renderer, { minCandidates: this._occlusionMinCandidates });
        if (gpuTier.supported()) {
          this._occlusionTier = gpuTier;
        } else {
          // WebGPU tier construction reported unsupported on a renderer that
          // DID pass the isWebGPURenderer check -- this can happen if the
          // backend fell back internally after construction. Do NOT fall
          // back to OcclusionQueryTier here: it calls renderer.getContext()
          // expecting a WebGL2RenderingContext, which a WebGPURenderer does
          // not expose the same way, and its own supported()===false would
          // then permanently disable _useOcclusionQuery for a renderer that
          // might still recover. Leave _occlusionTier null and retry next
          // frame instead (matches the "no tier yet" fail-open contract).
        }
      }).catch((err) => {
        this._webgpuTierLoading = false;
        if (typeof console !== 'undefined') console.warn('[model-pool] webgpu-hiz-tier.js dynamic import failed:', err);
        // Import failed (e.g. the page's importmap has no 'three/tsl').
        // Only fall back to the WebGL2 OcclusionQueryTier if the renderer
        // is NOT a WebGPURenderer -- on an actual WebGPURenderer that path
        // cannot work (see note above) and would just disable occlusion
        // culling entirely via a false supported() reading.
        if (this._useOcclusionQuery && !this._occlusionTier && this.renderer && !this.renderer.isWebGPURenderer) {
          this._occlusionTier = new OcclusionQueryTier(this.renderer, { minCandidates: this._occlusionMinCandidates });
          if (!this._occlusionTier.supported()) { this._occlusionTier = null; this._useOcclusionQuery = false; }
        }
      });
      return null; // this frame: no tier yet, caller no-ops (fail-open)
    }
    // WebGPURenderer path: either the dynamic import above is still in
    // flight (this._webgpuTierLoading true — return null, fail-open this
    // frame, the .then()/.catch() will assign a tier once it settles) or it
    // already failed/produced null for a reason logged there. Either way,
    // NEVER fall through to constructing OcclusionQueryTier against a
    // WebGPURenderer below -- that tier's constructor calls
    // renderer.getContext() expecting a WebGL2RenderingContext, which a
    // WebGPURenderer doesn't provide the same way, so supported() reads
    // false and the bug this fixes is: that false permanently disabled
    // _useOcclusionQuery even though the REAL WebGPU tier might still load
    // successfully a moment later.
    if (this.renderer && this.renderer.isWebGPURenderer) return this._occlusionTier;
    if (!this._occlusionTier) {
      if (!this.renderer) return null;
      // Plain WebGLRenderer (the common case): WebGL2 occlusion-query tier,
      // no dynamic import needed -- it's a normal static dependency of this
      // file already.
      this._occlusionTier = new OcclusionQueryTier(this.renderer, { minCandidates: this._occlusionMinCandidates });
      if (!this._occlusionTier.supported()) {
        // No WebGL2 occlusion query support (e.g. WebGL1 fallback context) —
        // degrade to frustum-only culling silently, never a hard failure.
        this._occlusionTier = null;
        this._useOcclusionQuery = false;
      }
    }
    return this._occlusionTier;
  }

  _getImpostorTier() {
    if (!this._useImpostorFinalLod) return null;
    if (!this._impostorTier) {
      if (!this.renderer || !this.scene) return null;
      this._impostorTier = this._useImpostorEz
        ? new OctahedralImpostorEzTier(this.renderer, {
            grid: this._impostorGrid,                 // sprites per atlas side
            textureSize: this._impostorTextureSize,   // 1024 -> ~1M-tri source models
            maxImpostorAssets: this._impostorMaxAssets,
            useHemiOctahedron: this._impostorHemiOcta,
          })
        : new OctahedralImpostorTier(this.renderer, {
            grid: this._impostorGrid,
            cellPx: this._impostorCellPx,
            blend: this._impostorBlend,
            padding: this._impostorPadding,
          });
      this.scene.add(this._impostorTier.mesh);
    }
    return this._impostorTier;
  }

  _getInstancedSlot(asset, meshDescIdx, lodIdx) {
    const desc = asset.meshLodDescs[meshDescIdx];
    if (!desc) return null;
    const lod = desc.lods[lodIdx];
    if (!lod || (lod.kind || 'textured') !== 'unskinned') return null;
    const key = `${asset.url}|${meshDescIdx}|${lodIdx}`;
    let slot = this._instancedSlots.get(key);
    if (slot) return slot;
    const geo = asset.geoCache.get(`${desc.meshIndex}:${desc.primIndex}:${lodIdx}`);
    if (!geo) return null; // not loaded yet

    // BatchedMesh FAR tier: one shared BatchedMesh draws ALL distinct far
    // geometries in ~1 draw call (vs one InstancedBatch/draw per distinct
    // asset). Return a per-(asset,lod) adapter that delegates to the shared
    // tier; the entity code uses it exactly like an instanced slot. LOD changes
    // within the far tier become synchronous setGeometryIdAt swaps.
    if (this._useBatchedFarTier) {
      if (!this._batchedFarTier) {
        this._batchedFarTier = new BatchedFarTier(this);
        this.scene.add(this._batchedFarTier.mesh);
      }
      const adapter = this._batchedFarTier.slotAdapter(asset, meshDescIdx, lodIdx, geo);
      this._instancedSlots.set(key, adapter);
      return adapter;
    }

    // Phase 3 QW4: Check if this geometry can reuse an existing InstancedSlot
    // from another asset (geometry-hash based reuse)
    const reusableSlot = this._findReusableSlotByGeometryHash(geo);
    if (reusableSlot) {
      this._instancedSlots.set(key, reusableSlot);
      return reusableSlot; // Reuse existing slot instead of creating new one
    }

    // Material Grouping: Use global FAR-tier material if enabled
    let mat;
    if (this._globalMaterialPool._useGlobalMaterialPool) {
      mat = this._globalMaterialPool.getMaterialForTier('far');
    } else {
      // Fallback: create per-LOD material (baseline behavior)
      mat = new THREE.MeshLambertMaterial({ vertexColors: true });
      mat.onBeforeCompile = (shader) => {
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <color_fragment>',
          `#if defined( USE_COLOR_ALPHA )
            diffuseColor.rgb *= pow(vColor.rgb, vec3(2.2));
            diffuseColor.a *= vColor.a;
          #elif defined( USE_COLOR )
            diffuseColor.rgb *= pow(vColor, vec3(2.2));
          #endif`
        );
      };
    }

    slot = new InstancedSlot(this, asset, meshDescIdx, lodIdx, geo, mat);
    this._instancedSlots.set(key, slot);

    // Phase 3 QW4: Register this geometry for potential future reuse
    this._registerGeometryHashSlot(geo, slot);

    // Attach the instanced mesh to the same scene as entities.
    this.scene.add(slot.mesh);
    return slot;
  }

  // Get-or-load an asset; idempotent.
  async _resolveAsset(url) {
    let a = this._assets.get(url);
    if (!a) {
      a = new Asset(this, url);
      this._assets.set(url, a);
      a.ready.then(() => this.emit('asset-ready', a)).catch((e) => this.emit('asset-error', { asset: a, error: e }));
    }
    return a;
  }

  // Spawn an entity from a URL. Returns an Entity handle synchronously; the
  // Entity emits 'ready' once loading completes.
  spawn(url, opts = {}) {
    if (!url) throw new Error('spawn(): url required');
    if (typeof url !== 'string') throw new TypeError(`spawn(): url must be a string, got ${typeof url}`);
    const assetPromise = this._resolveAsset(url);
    // Build a placeholder entity tied to a yet-to-resolve asset.
    // We attach to the entity once the asset is ready inside Entity._bootstrap.
    const placeholder = { _disposed: false };
    let actualEntity = null;
    // Wrap into a lightweight proxy so caller can listen on events even
    // before bootstrap finishes.
    const proxy = new Emitter();
    proxy.root = new THREE.Object3D();
    proxy.root.name = `pending_${++this._nextEntityId}`;
    proxy.dispose = () => {
      placeholder._disposed = true;
      if (actualEntity) actualEntity.dispose();
      else proxy.root.parent?.remove(proxy.root);
    };
    assetPromise.then((asset) => {
      if (placeholder._disposed) return;
      actualEntity = new Entity(this, asset, opts);
      this._entities.add(actualEntity);
      // Stitch: replace proxy.root with actualEntity.root in the parent.
      const parent = proxy.root.parent;
      if (parent) {
        parent.add(actualEntity.root);
        parent.remove(proxy.root);
      }
      // Re-forward events.
      actualEntity.on('ready', (e) => proxy.emit('ready', e));
      actualEntity.on('lod-changed', (e) => proxy.emit('lod-changed', e));
      actualEntity.on('disposed', (e) => proxy.emit('disposed', e));
      actualEntity.on('error', (e) => proxy.emit('error', e));
      // Expose useful entity props on the proxy.
      proxy.actualEntity = actualEntity;
      proxy.root = actualEntity.root;
    }).catch((e) => proxy.emit('error', e));
    return proxy;
  }

  // Per-frame update: call from your render loop AFTER advancing camera.
  update() {
    const tUpdate0 = performance.now();
    const now = tUpdate0;
    const dt = (now - this._lastTick) / 1000;
    this._lastTick = now;
    // Drain the on-the-fly impostor bake queue (<=budget/frame). Each atlas bake
    // is GRIDxGRID synchronous renders (~tens of ms); the cap spreads them across
    // frames so a far camera bringing many assets into range at once never
    // stutters — entities awaiting their atlas stay on their existing far LOD.
    if (this._useImpostorFinalLod) this._drainImpostorBakes();
    // EMA FPS over ~500ms (faster feedback than 1s).
    const instFps = dt > 0 ? 1 / dt : 60;
    this._fpsEma = this._fpsEma * 0.85 + instFps * 0.15;

    // Push the monotonic clock to the GPU-lerp shader once per frame — the ONLY
    // per-frame CPU->GPU write for far entities interpolating on the GPU. (The
    // far tier reads it via the uNow uniform; entities in flight need no matrix
    // write at all.)
    if (this._batchedFarTier) this._batchedFarTier.updateNow(now / 1000);

    // Interpolate active position targets BEFORE the entity/LOD pass so lerped
    // positions feed this frame's distance + LOD picks. O(moving entities).
    // (CPU fallback path for hero/mid tiers; far entities use the GPU lerp above.)
    this._drainMovers(now);

    // Staticness count is folded into the single entity pass below to avoid a
    // second full walk of every entity per frame (was a separate loop). The
    // frustum-check interval is driven by LAST frame's count — a one-frame lag
    // on a smoothing heuristic, harmless and already implicit in the EMA.
    const totalEntities = this._entities.size;
    const prevMoving = this._lastFrameMovingCount || 0;
    const sceneStaticnessPercent = totalEntities > 0 ? ((totalEntities - prevMoving) / totalEntities) * 100 : 100;
    this._dynamicFrustumCheckInterval = prevMoving === 0 ? 10 : (sceneStaticnessPercent >= 95 ? 8 : 5);
    let movingCount = 0; // accumulated in the entity pass below
    // Adaptive budget control: react to sustained low FPS.
    const target = this.targetFps;
    const entityCount = this._entities.size;
    // Frequency scales with load: many entities → faster response (more aggressive for 140 FPS target)
    const adjustFreq = entityCount > 500 ? 6 : entityCount > 200 ? 10 : 20;
    // Knob magnitude also scales: more entities = more aggressive tightening
    const knobStep = entityCount > 500 ? 30 : 20;
    const midPxMax = entityCount > 500 ? 250 : 200;

    // Closed-loop LOD-DISTANCE controller. Instead of clamping a global LOD
    // ceiling (which banged many entities across discrete tiers at once and
    // oscillated), the controller continuously scales _lodDistanceScale — a
    // multiplier on the per-entity screen-size used by _pickMeshLod. FPS below
    // target shrinks the scale so every model drops to its next-lower LOD a bit
    // sooner (as if slightly farther); FPS above target grows it so detail
    // extends out. Because it's a smooth continuous knob applied per entity by
    // distance, there's no tier stampede — each model crosses its own threshold
    // independently and gradually. The ceiling is left fixed (no FPS clamp).
    //
    // Stability: a dead-band around the target stops adjustment once FPS is
    // close, and the per-frame step is small + rate-limited so the scene eases
    // to a resting scale rather than hunting.
    if (this._lodDistanceScale == null) this._lodDistanceScale = 1;
    const lowBand = target - Math.max(6, target * 0.06);
    const highBand = target + Math.max(8, target * 0.10);
    const SCALE_MIN = 0.15, SCALE_MAX = 1.6;
    // Small proportional step per adjust tick, scaled by how far off we are.
    if (!this._lodAdjustCountdown) this._lodAdjustCountdown = 4;
    if (--this._lodAdjustCountdown <= 0) {
      this._lodAdjustCountdown = 4;
      if (this._fpsEma < lowBand && this._lodDistanceScale > SCALE_MIN) {
        // Below target → pull LODs nearer (cheaper). Step grows with the deficit.
        const deficit = Math.min(1, (lowBand - this._fpsEma) / Math.max(1, lowBand));
        this._lodDistanceScale = Math.max(SCALE_MIN, this._lodDistanceScale * (1 - 0.06 - 0.1 * deficit));
        this.emit('budget-adjust', { reason: 'fps-low-loddist', lodDistanceScale: this._lodDistanceScale, fps: this._fpsEma });
      } else if (this._fpsEma > highBand && this._lodDistanceScale < SCALE_MAX) {
        // Above target with headroom → push LODs out (more detail), gently.
        this._lodDistanceScale = Math.min(SCALE_MAX, this._lodDistanceScale * 1.04);
        this.emit('budget-adjust', { reason: 'fps-high-loddist', lodDistanceScale: this._lodDistanceScale, fps: this._fpsEma });
      }
    }

    // Interval-based adjustments for midPx and heroCap (smooth, gradual changes).
    if (!this._fpsAdjustCountdown) this._fpsAdjustCountdown = adjustFreq;
    this._fpsAdjustCountdown--;
    if (this._fpsAdjustCountdown <= 0) {
      this._fpsAdjustCountdown = adjustFreq;
      if (this._fpsEma < target - 5) {
        // Sustained low FPS. Tighten midPx/heroCap with load-aware aggressiveness.
        let changed = false;
        if (this.midPx < midPxMax) {
          this.midPx = Math.min(midPxMax, this.midPx + knobStep);
          changed = true;
        } else if (this.heroCap > 5) {
          this.heroCap = Math.max(5, this.heroCap - 5);
          changed = true;
        }
        if (changed) this.emit('budget-adjust', {
          reason: 'fps-sustained-low', midPx: this.midPx, heroCap: this.heroCap, fps: this._fpsEma,
        });
      } else if (this._fpsEma > target + 5) {
        // Headroom — relax knobs in reverse priority.
        let changed = false;
        if (this.heroCap < 20) {
          this.heroCap = Math.min(20, this.heroCap + 5);
          changed = true;
        } else if (this.midPx > 50) {
          this.midPx = Math.max(50, this.midPx - 10);
          changed = true;
        }
        if (changed) this.emit('budget-adjust', {
          reason: 'fps-headroom', midPx: this.midPx, heroCap: this.heroCap, fps: this._fpsEma,
        });
      }
    }
    // Build frustum once per frame.
    const tFrustum0 = performance.now();
    this.camera.updateMatrixWorld();
    this._tmpMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._tmpMatrix);
    // PHASE 2 OPTIMIZATION: Update cached frustum planes once per frame
    // This eliminates per-vertex plane extraction (16 instructions per vertex).
    if (this._frustumCache) {
      // Pass the projView-derived frustum already built above — updatePlanes
      // would otherwise redo the identical projection*matrixWorldInverse
      // multiply + THREE.Frustum.setFromProjectionMatrix extraction a second
      // time per frame for the same camera/matrix.
      this._frustumCache.updatePlanes(this.camera, this._frustum);
    }
    // Publish projView to every InstancedSlot's shader uniform so the GPU
    // can run the per-instance frustum cull pass.
    // Cache tan(fov/2) once per frame; Entity._update reads this instead of
    // recomputing degToRad+tan per entity.
    this._fovTanHalf = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    // Detect camera movement so static entities recompute screenPx/LOD when the
    // view changes (orbit, zoom, fly-through). Compare position + fov to last
    // frame with a small epsilon to avoid thrash when the camera is still.
    {
      const cp = this.camera.position;
      const lp = this._lastCamPos || (this._lastCamPos = { x: Infinity, y: 0, z: 0, fov: 0, qx: 0, qy: 0, qz: 0, qw: 1 });
      const cq = this.camera.quaternion;
      // Quaternion dot-product delta: catches a PURE in-place rotation (no
      // position/fov change) that would otherwise never bump _cameraMoved —
      // there is no periodic/time-based fallback recheck elsewhere in this
      // file, so a static instanced entity's cached screenPx/LOD would
      // otherwise never re-evaluate while only the view direction changes.
      // 1 - |dot| ~= (theta/2)^2 for small angles; ~0.0005 corresponds to a
      // rotation of roughly half a degree, well below visible LOD-pop thresholds
      // but far above quaternion normalization jitter.
      const qDot = cq.x * lp.qx + cq.y * lp.qy + cq.z * lp.qz + cq.w * lp.qw;
      const rotated = (1 - Math.abs(qDot)) > 0.0005;
      const moved = Math.abs(cp.x - lp.x) > 1e-3 || Math.abs(cp.y - lp.y) > 1e-3 || Math.abs(cp.z - lp.z) > 1e-3 || this.camera.fov !== lp.fov || rotated;
      this._cameraMoved = moved;
      if (moved) { lp.x = cp.x; lp.y = cp.y; lp.z = cp.z; lp.fov = this.camera.fov; lp.qx = cq.x; lp.qy = cq.y; lp.qz = cq.z; lp.qw = cq.w; }
      // Bump the LOD-evaluation epoch when the camera moved OR the global LOD
      // distance-scale shifted beyond a quantum since last frame. Entities only
      // re-pick their LOD when the epoch changes (see Entity._update), so a
      // still camera + stable scale produces zero per-frame LOD churn. Quantize
      // the scale comparison so the controller's tiny proportional steps don't
      // count as a change (that scale-hunting was the still-camera stutter).
      if (this._lodEpoch == null) { this._lodEpoch = 1; this._lastLodScale = this._lodDistanceScale; this._cameraMoved = true; }
      const scaleNow = this._lodDistanceScale || 1;
      const scaleChanged = Math.abs(scaleNow - (this._lastLodScale ?? scaleNow)) > 0.04;
      if (moved || scaleChanged) { this._lodEpoch++; this._lastLodScale = scaleNow; }
    }
    const vh = this.renderer.domElement.clientHeight;
    // PERF: every InstancedSlot's projViewMatrix uniform receives the SAME
    // projView matrix this frame (one camera, one pool-wide `update()` per
    // frame) — there is no per-slot variation. Previously each slot did a
    // full mat4 .copy() of `this._tmpMatrix` into its own Matrix4 (N copies
    // for N slots); now every slot's uniform.value is repointed to the SAME
    // shared matrix reference, collapsing N copies to N cheap reference
    // assignments. Safe because three.js reads uniform.value synchronously
    // during this frame's render (after this loop, before `_tmpMatrix` is
    // ever mutated again next frame's update()), and a `{value}` uniform
    // wrapper object's IDENTITY must stay stable across frames (captured by
    // reference at shader compile time via `shader.uniforms.projViewMatrix =
    // uniforms.projViewMatrix`) but `.value` itself may be reassigned freely.
    // Note (documented, not fixed here): `_instancedSlots` can map multiple
    // keys to the SAME underlying slot object (QW4 reuse pattern), so this
    // loop may touch one physical slot more than once per frame — each extra
    // touch is now a no-op-cost reference reassignment rather than a mat4
    // copy, so it is left as a minor, low-value dedup opportunity.
    for (const slot of this._instancedSlots.values()) {
      // BatchedFarTier adapters have no projViewMatrix uniform — BatchedMesh
      // does its own transform + per-instance cull internally.
      if (slot._uniforms) slot._uniforms.projViewMatrix.value = this._tmpMatrix;
    }
    const tFrustum1 = performance.now();

    // VRAM-aware dynamic pool sizing: monitor GPU memory usage and adjust dynamically
    const estimatedVramBytes = this._estimatedVramMB * 1024 * 1024;
    this._vramRatioMonitor.currentRatio = estimatedVramBytes > 0 ? this._totalBytes / estimatedVramBytes : 0;
    this._vramRatioMonitor.peakRatio = Math.max(this._vramRatioMonitor.peakRatio, this._vramRatioMonitor.currentRatio);

    // Decrement cooldown timer for dynamic VRAM adjustments
    if (this._vramRatioMonitor.adjustmentCooldown > 0) {
      this._vramRatioMonitor.adjustmentCooldown--;
    }

    // Critical path: if VRAM ratio exceeds 70%, aggressively reduce LOD ceiling and entity count
    if (this._vramRatioMonitor.currentRatio > this._vramThresholdCritical) {
      if (this._vramRatioMonitor.adjustmentCooldown === 0) {
        // Reduce LOD ceiling to force more entities to FAR tier (lower quality, but GPU-instanced)
        const nextCeil = (this._currentCeilingLod ?? 5) - 1;
        if (nextCeil >= 0 && this._currentCeilingLod !== nextCeil) {
          this._currentCeilingLod = nextCeil;
          console.warn(`[VRAM] ratio ${(this._vramRatioMonitor.currentRatio * 100).toFixed(1)}% > ${(this._vramThresholdCritical * 100).toFixed(0)}% — reduced LOD ceiling to ${nextCeil}`);
          this.emit('vram-critical', {
            ratio: this._vramRatioMonitor.currentRatio,
            bytes: this._totalBytes,
            estimatedVramMB: this._estimatedVramMB,
            action: 'reduce-lod-ceiling',
          });
        }
        // Also aggressively increase midPx to push more entities to FAR tier
        if (this.midPx < 250) {
          const oldMidPx = this.midPx;
          this.midPx = Math.min(250, this.midPx + 20);
          console.warn(`[VRAM] increased midPx from ${oldMidPx} to ${this.midPx} to reduce VRAM pressure`);
        }
        this._vramRatioMonitor.adjustmentCooldown = 120; // ~2s at 60 FPS — long cooldown avoids ceiling oscillation
        this._vramRatioMonitor.lastAdjustmentRatio = this._vramRatioMonitor.currentRatio;
      }
    }
    // Warning path: if VRAM ratio exceeds 65%, start warming user
    else if (this._vramRatioMonitor.currentRatio > this._vramThresholdWarning && this._vramRatioMonitor.currentRatio <= this._vramThresholdCritical) {
      if (this._vramRatioMonitor.adjustmentCooldown === 0) {
        console.warn(`[VRAM] warning: ratio ${(this._vramRatioMonitor.currentRatio * 100).toFixed(1)}% — approaching critical threshold`);
        this.emit('vram-warning', {
          ratio: this._vramRatioMonitor.currentRatio,
          bytes: this._totalBytes,
          estimatedVramMB: this._estimatedVramMB,
        });
        this._vramRatioMonitor.adjustmentCooldown = 60;
      }
    }
    // Safe path: VRAM no longer AUTO-RELAXES the ceiling. This was the ~2s LOD-pop
    // root cause: lowering the ceiling pushed entities to the cheap FAR/instanced
    // tier, which dropped _totalBytes below the safe threshold; the safe path then
    // raised the ceiling back up, textured LODs reloaded, bytes climbed back over
    // critical, and the cycle repeated with the ~120-frame (~2s) cooldown setting
    // the period. The VRAM action fed straight back into its own trigger — a limit
    // cycle no cooldown can break. Fix: VRAM is a one-way RATCHET. It only clamps
    // the ceiling DOWN under genuine memory pressure; relaxation is owned solely by
    // the FPS controller above (which has proper sustained-headroom hysteresis and
    // will raise the ceiling when FPS proves there's room). The two controllers no
    // longer write the ceiling in opposite directions, so they can't tug-of-war.

    // Reset budget tracking for this frame
    this._heroFrameTimeMs = 0;
    this._midFrameTimeMs = 0;
    let visible = 0;

    // First pass: update every entity ONCE and, in the same walk:
    //  - accumulate movingCount (staticness — was its own loop)
    //  - mark unload-manager visibility (was its own loop after tiering)
    //  - collect visible entities into REUSED parallel arrays (no per-frame
    //    object-literal allocation — the old code pushed {entity,distance,...}
    //    objects every frame, churning the GC in the hot path).
    // Parallel arrays grow once and are reused; only `_distCount` resets/frame.
    let ents = this._distEntities;
    let dists = this._distValues;
    let order = this._distOrder;
    const cap = this._entities.size;
    if (!ents || ents.length < cap) {
      ents = this._distEntities = new Array(cap);
      dists = this._distValues = new Float64Array(cap);
      order = this._distOrder = new Int32Array(cap);
    }
    let n = 0;
    const stream = this._enableDeferredStreaming;
    if (stream) this._lodUnloadManager.resetVisibility();
    // Occlusion tier (opt-in): entities its LAST resolved query found fully
    // hidden are skipped here before frustum/LOD work runs — same shape as
    // the frustum early-return above, so an occluded entity costs almost
    // nothing per frame beyond the map lookup. Occlusion is a strict subset
    // of "invisible" (an occluded entity is necessarily off-screen from the
    // viewer's perspective too), so gating before _update is correct: the
    // entity's OWN root.visible stays whatever frustum culling would have
    // set, but we skip its LOD/distance bookkeeping entirely when occluded.
    const occTier = this._useOcclusionQuery ? this._getOcclusionTier() : null;
    this._occlusionCandidates = this._occlusionCandidates || [];
    this._occlusionCandidates.length = 0;
    for (const e of this._entities) {
      if (e._disposed) continue;
      if (occTier && occTier.isOccluded(e)) {
        if (e.root.visible) e.root.visible = false;
        if (stream) this._lodUnloadManager.markInvisible(e);
        // MUST stay a query candidate even while hidden: the occluding
        // geometry can move/disappear next frame (a wall sliding away, the
        // occluder itself getting culled), and this is the ONLY path that
        // re-tests and un-hides it. Dropping it from candidates here would
        // make occlusion a one-way, permanent hide -- worse than no culling
        // at all. The candidate test itself is a cheap bounding-box query,
        // not the entity's full LOD/distance/animation work, which IS still
        // skipped below via continue.
        this._occlusionCandidates.push(e);
        continue; // fully occluded last frame — skip LOD/distance work this frame
      }
      if (e.root.matrixAutoUpdate || e._boundDirty) movingCount++;
      const result = e._update(this.camera, vh, dt, this._currentCeilingLod, this._frustum, this.animationThrottleDistance);
      const { distance, screenPx } = result;
      if (screenPx > 0) {
        visible++;
        if (occTier) this._occlusionCandidates.push(e);
      }
      if (distance < Infinity) {
        ents[n] = e; dists[n] = distance; n++;
      }
      if (stream) {
        if (e.root.visible) this._lodUnloadManager.markVisible(e);
        else this._lodUnloadManager.markInvisible(e);
      }
    }
    this._distCount = n;
    this._lastFrameMovingCount = movingCount; // drives next frame's staticness interval

    // Sort the index array by distance (closest first) — comparator reads the
    // typed dists[] rather than dereferencing wrapper objects. Tier allocation
    // is greedy-by-proximity, so closest-first order must be preserved.
    // SKIP the O(N log N) sort when nothing that affects ordering changed this
    // frame: camera still, no movers, same visible set. Distances are constant
    // for static entities under a still camera, so last frame's order holds.
    const sortOrder = order.subarray(0, n);
    const orderStable = !this._cameraMoved && movingCount === 0 && n === this._lastSortN;
    if (!orderStable) {
      for (let i = 0; i < n; i++) order[i] = i; // reset to identity, then sort
      sortOrder.sort((a, b) => dists[a] - dists[b]);
      this._lastSortN = n;
    }

    // Second pass: allocate entities to tiers by distance + per-frame budgets.
    let hero = 0, mid = 0, far = 0;
    for (let k = 0; k < n; k++) {
      const idx = sortOrder[k];
      const entity = ents[idx];
      const distance = dists[idx];
      let assignedTier;
      if (distance < this._heroDist && this._heroFrameTimeMs < this._heroBudgetMs) {
        assignedTier = 'hero';
        this._heroFrameTimeMs += 0.15; // ~0.15ms per entity (animation + per-entity draw)
        hero++;
      } else if (distance < this._midDist && this._midFrameTimeMs < this._midBudgetMs) {
        assignedTier = 'mid';
        this._midFrameTimeMs += 0.08; // ~0.08ms per entity (decimated geometry, cheaper material)
        mid++;
      } else {
        assignedTier = 'far';
        far++;
      }
      entity._assignedTier = assignedTier;
    }

    // Third pass: dynamic distance threshold rebalancing based on budget utilization
    // This ensures budgets stay within target range while allowing expansion when headroom exists
    const heroUtilization = this._heroBudgetMs > 0 ? this._heroFrameTimeMs / this._heroBudgetMs : 0;
    const midUtilization = this._midBudgetMs > 0 ? this._midFrameTimeMs / this._midBudgetMs : 0;

    // HERO tier rebalancing
    if (heroUtilization > 1.1) {
      // Budget exceeded by >10%: aggressively shrink HERO distance to reduce HERO tier population
      this._heroDist = Math.max(5, this._heroDist - 2);
    } else if (heroUtilization < 0.5 && hero > 0) {
      // Budget underutilized (<50%): expand HERO distance to allow more entities in HERO tier
      this._heroDist = Math.min(50, this._heroDist + 1);
    }

    // MID tier rebalancing
    if (midUtilization > 1.1) {
      // Budget exceeded by >10%: shrink MID distance to reduce MID tier population
      this._midDist = Math.max(20, this._midDist - 3);
    } else if (midUtilization < 0.5 && mid > 0) {
      // Budget underutilized (<50%): expand MID distance to allow more entities in MID tier
      this._midDist = Math.min(100, this._midDist + 2);
    }
    // Flush all dirty instance matrices to GPU once per frame.
    for (const slot of this._instancedSlots.values()) {
      slot.flushMatrixUpdates();
    }
    // Flush the octahedral impostor tier's dirty instance-matrix ranges (see
    // OctahedralImpostorTier.flush — narrows the upload to only the instances
    // acquired/moved/released this frame instead of the whole maxInstances buffer).
    if (this._impostorTier) this._impostorTier.flush();
    const tEntities1 = performance.now();

    // Asset Streaming: visibility was already marked in the single entity pass
    // above (resetVisibility + markVisible/markInvisible). Here we only run the
    // periodic unload scan.
    if (this._enableDeferredStreaming) {
      // Periodically scan for unloads (every 5 frames to avoid overhead)
      if (!this._unloadScanCounter) this._unloadScanCounter = 5;
      this._unloadScanCounter--;
      if (this._unloadScanCounter <= 0) {
        this._lodUnloadManager.scanForUnload(this._assets, this._totalBytes);
        this._unloadScanCounter = 5;
      }
    }

    // Update tier distribution stats. Raw integer counts every frame (cheap).
    this._stats.hero = hero;
    this._stats.mid = mid;
    this._stats.far = far;
    this._stats.heroBudgetTarget = this._heroBudgetMs;
    this._stats.midBudgetTarget = this._midBudgetMs;

    // Cosmetic stats (toFixed/parseFloat conversions + the tierSummary template
    // literal) are HUD-only and don't need per-frame precision. Rebuild them at
    // most every 6 frames to keep ~8 string<->number conversions + a template
    // literal out of the per-frame hot path. Raw numeric fields above stay live.
    this._statsCosmeticCountdown = (this._statsCosmeticCountdown || 0) - 1;
    if (this._statsCosmeticCountdown <= 0) {
      this._statsCosmeticCountdown = 6;
      this._stats.heroBudgetMs = parseFloat(this._heroFrameTimeMs.toFixed(2));
      this._stats.midBudgetMs = parseFloat(this._midFrameTimeMs.toFixed(2));
      this._stats.heroDist = parseFloat(this._heroDist.toFixed(1));
      this._stats.midDist = parseFloat(this._midDist.toFixed(1));
      this._stats.tierSummary = `HERO: ${hero}/${this._heroCap} (${this._heroFrameTimeMs.toFixed(1)}ms), MID: ${mid} (${this._midFrameTimeMs.toFixed(1)}ms), FAR: ${far} (0ms)`;
    }

    this._stats.msFrustum = tFrustum1 - tFrustum0;
    this._stats.msEntities = tEntities1 - tFrustum1;
    // Maintain byte budget. If still over budget after eviction (because
    // active LODs can't be evicted), tighten midPx so more entities drop
    // to the unskinned tier — that releases higher-LOD references and
    // unlocks eviction next sweep. Geometry shape is still preserved
    // because the unskinned tier is a real (decimated) mesh, not a sprite.
    const tBudget0 = performance.now();
    this._enforceBudget();
    const tBudget1 = performance.now();
    // Frame-budgeted, headroom-gated warm loading of wanted-but-unresident LODs
    // (network-lazy, GPU-eager, piecemeal) so LOD switches never pay a fetch /
    // decode / first-use upload on the switch frame.
    this._drainLodWarm();
    this._drainGpuWarm(); // at most one synchronous GPU upload per frame
    if (this._totalBytes > this.byteBudget && this.midPx < 200) {
      // Larger midPx → wider FAR catch (entities up to midPx screen-size).
      this.midPx = Math.min(200, this.midPx + 5);
      this.emit('budget-adjust', { reason: 'over-budget', midPx: this.midPx, bytes: this._totalBytes, budget: this.byteBudget });
    }
    // Additional stats tracking
    this._stats.fps = this._fpsEma;
    this._stats.entities = this._entities.size;
    this._stats.visible = visible;
    this._stats.drawCalls = this.renderer.info?.render?.calls ?? 0;
    this._stats.ceilingLod = this._currentCeilingLod;
    this._stats.bytes = this._totalBytes;
    this._stats.assets = this._assets.size;
    this._stats.inFlight = this._inFlight;
    this._stats.msBudget = tBudget1 - tBudget0;
    this._stats.msTotal = performance.now() - tUpdate0;
    this._stats.midPx = this.midPx;
    this._stats.heroCap = this.heroCap;
    // VRAM stats for HUD display
    this._stats.vram = {
      currentRatio: this._vramRatioMonitor.currentRatio,
      peakRatio: this._vramRatioMonitor.peakRatio,
      estimatedVramMB: this._estimatedVramMB,
      usedMB: this._totalBytes / (1024 * 1024),
    };
    this.emit('fps', this._stats);
  }

  // Public stats accessor (cheap, no allocation).
  getStats() {
    const stats = this._stats;
    stats.impostors = this._impostorActiveCount;
    stats.impostorLayers = this._impostorTier ? this._impostorTier._nextLayer : 0;
    // The four sub-stats (each allocates a fresh object) are HUD diagnostics that
    // change slowly; refresh them at most every 6th call instead of every call to
    // avoid per-call allocation churn. The base this._stats fields are live.
    if ((this._subStatsCountdown = (this._subStatsCountdown || 0) - 1) <= 0) {
      this._subStatsCountdown = 6;
      if (this._globalMaterialPool) stats.materialPooling = this._globalMaterialPool.getStats();
      if (this._deferredLoadQueue) stats.deferredLoading = this._deferredLoadQueue.getStats();
      if (this._lodUnloadManager) stats.unloadManager = this._lodUnloadManager.getStats();
      if (this._occlusionTier) stats.occlusion = this._occlusionTier.stats;
    }
    return stats;
  }

  // --- Byte tracking + budget ---------------------------------------------
  _trackBytes(assetUrl, url, bytes) {
    this._totalBytes += bytes;
    let log = this._byteLog.get(assetUrl);
    if (!log) { log = new Map(); this._byteLog.set(assetUrl, log); }
    log.set(url, bytes);
  }
  _untrackBytes(assetUrl, url) {
    const log = this._byteLog.get(assetUrl);
    if (!log) return;
    const b = log.get(url) || 0;
    this._totalBytes -= b;
    log.delete(url);
  }
  _enforceBudget() {
    if (this._totalBytes <= this.byteBudget) return;
    // Throttle the expensive scan body (not the byteBudget gate above): once
    // over budget in a steady state (common — the budget line is a moving
    // target the FPS/VRAM controllers hover near), this function used to
    // rebuild a full `inUse` Set via a per-entity x per-trackedMesh walk (with
    // template-literal key allocation per mesh/tex) on EVERY frame. That's the
    // one per-frame GC/CPU cost in update() with no cooldown, unlike the
    // sibling unload-scan (every 5 frames) and stats-cosmetics (every 6
    // frames) throttles right next to it. A still-over-budget scene pays this
    // full O(entities) scan+allocation every frame for no new information —
    // eviction candidates don't change frame-to-frame once the in-use set is
    // stable. Run at most every N frames; still re-checked every frame via the
    // cheap byteBudget comparison above so a NEWLY over-budget frame is never
    // silently skipped for more than the cooldown window.
    if (!this._enforceBudgetCounter) this._enforceBudgetCounter = 0;
    if (this._enforceBudgetCounter > 0) { this._enforceBudgetCounter--; return; }
    this._enforceBudgetCounter = 5;
    // Find evict candidates: LODs no current entity is using AND not inline.
    // Walk every asset's cached non-inline geo/tex; drop any whose key isn't
    // currently active on any entity.
    const inUse = new Set();
    for (const e of this._entities) {
      for (const tm of e.trackedMeshes) {
        const d = e.asset.meshLodDescs[tm.meshDescIdx];
        if (d) inUse.add(`${e.asset.url}|${d.meshIndex}:${d.primIndex}:${tm.currentLod}`);
        for (let ti = 0; ti < tm.texState.length; ti++) {
          const td = e.asset.texLodDescs[ti];
          if (td) inUse.add(`${e.asset.url}|tex:${td.textureIndex}:${tm.texState[ti].currentLod}`);
        }
      }
    }
    let evicted = 0;
    for (const asset of this._assets.values()) {
      for (const desc of asset.meshLodDescs) {
        for (let li = 0; li < desc.lods.length; li++) {
          if (desc.lods[li].inline) continue;
          const key = `${asset.url}|${desc.meshIndex}:${desc.primIndex}:${li}`;
          if (!inUse.has(key)) {
            if (asset.evictMeshLod(asset.meshLodDescs.indexOf(desc), li)) {
              this._totalBytes -= (desc.lods[li].bytes || 0);
              evicted++;
            }
          }
        }
      }
      for (const desc of asset.texLodDescs) {
        for (let li = 0; li < desc.lods.length; li++) {
          if (desc.lods[li].inline) continue;
          const key = `${asset.url}|tex:${desc.textureIndex}:${li}`;
          if (!inUse.has(key)) {
            if (asset.evictTexLod(asset.texLodDescs.indexOf(desc), li)) {
              this._totalBytes -= (desc.lods[li].bytes || 0);
              evicted++;
            }
          }
        }
      }
      if (this._totalBytes <= this.byteBudget) break;
    }
    if (evicted) this.emit('budget-pressure', { evicted, total: this._totalBytes, budget: this.byteBudget });

    // Dynamic budget adjustment: if memory ratio is unsafe, reduce budget; if safe for sustained duration, increase.
    const ratio = this._totalBytes / (this._estimatedVramMB * 1024 * 1024);
    if (ratio > 0.7) {
      // Memory pressure: reduce budget by 10% to stay safe.
      this.byteBudget = Math.floor(this.byteBudget * 0.9);
      this._budgetAdjustmentCooldown = 60; // prevent oscillation (60 frames = ~1 second at 60 FPS)
      this.emit('budget-warning', { ratio, newBudget: this.byteBudget, estimatedVramMB: this._estimatedVramMB });
    } else if (ratio < 0.4 && this._budgetAdjustmentCooldown === 0) {
      // Low utilization: accumulate frames with safe margin.
      this._budgetLowFrames++;
      if (this._budgetLowFrames >= 10) {
        // 10 consecutive frames below 40% utilization: safe to increase budget by 5%.
        const safeByteBudget = Math.floor((this._estimatedVramMB * 0.65) * 1024 * 1024);
        if (this.byteBudget < safeByteBudget) {
          this.byteBudget = Math.min(this.byteBudget * 1.05, safeByteBudget);
          this._budgetAdjustmentCooldown = 60;
          this._budgetLowFrames = 0;
          this.emit('budget-relaxed', { ratio, newBudget: this.byteBudget });
        }
      }
    } else {
      // Ratio back in safe zone: reset low-frame counter.
      this._budgetLowFrames = 0;
    }

    // Decrement cooldown timer.
    if (this._budgetAdjustmentCooldown > 0) {
      this._budgetAdjustmentCooldown--;
    }
  }

  // --- Bounded concurrent fetch queue --------------------------------------
  _enqueue(key, run) {
    const existing = this._loadQueue.get(key);
    if (existing) return existing;
    const p = new Promise((resolve, reject) => {
      const task = { key, run, resolve, reject };
      if (this._inFlight < this.maxConcurrentFetches) this._runTask(task);
      else this._pending.push(task);
    });
    this._loadQueue.set(key, p);
    p.finally(() => this._loadQueue.delete(key));
    return p;
  }
  async _runTask(task) {
    this._inFlight++;
    try {
      const r = await task.run();
      task.resolve(r);
    } catch (e) {
      task.reject(e);
    } finally {
      this._inFlight--;
      if (this._pending.length && this._inFlight < this.maxConcurrentFetches) {
        this._runTask(this._pending.shift());
      }
    }
  }

  // Tear the whole pool down: dispose every live entity, then every shared
  // asset (which deepDisposes any VRM runtime and frees cached GPU buffers).
  dispose() {
    // Idempotency guard (mirrors Entity.dispose()'s _disposed pattern): a second
    // call must be a safe no-op. Without this, re-disposing _impostorTier/
    // _batchedFarTier/_occlusionTier a second time hit already-freed WebGL
    // resources (render targets/materials) which three.js does not guarantee
    // is safe to dispose twice.
    if (this._disposed) return;
    this._disposed = true;
    for (const e of [...this._entities]) e.dispose();
    for (const asset of this._assets.values()) asset.dispose();
    this._assets.clear();
    this._loadQueue.clear();
    // Tear down the shared far/impostor tiers so their GPU buffers (the impostor
    // atlas array-texture is ~134 MB; the BatchedMesh holds multi-MB vertex/index
    // buffers) are freed instead of leaking on pool teardown.
    if (this._impostorTier) {
      if (this.scene && this._impostorTier.mesh) this.scene.remove(this._impostorTier.mesh);
      this._impostorTier.dispose();
      this._impostorTier = null;
    }
    if (this._batchedFarTier) {
      const m = this._batchedFarTier.mesh;
      if (this.scene && m) this.scene.remove(m);
      m?.geometry?.dispose?.();
      m?.material?.dispose?.();
      m?.dispose?.();
      this._batchedFarTier = null;
    }
  }
}
