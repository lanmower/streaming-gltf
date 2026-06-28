// octahedral-impostor-tier.js — the FINAL-LOD tier: every distant entity, across
// every distinct asset, collapses to ONE InstancedMesh draw of camera-facing
// billboards that sample a per-asset octahedral atlas held in a single WebGL2
// texture array (sampler2DArray). One asset == one array layer, baked on-the-fly
// the first time the asset reaches impostor distance.
//
// This sits one rung past batched-far-tier.js: that collapses N geometries into
// one BatchedMesh draw but still rasterizes ~hundreds of triangles per model;
// this collapses them to two triangles per model and a texture fetch. Interface
// mirrors BatchedFarTier (bakeAsset / acquire / release / setMatrix) so the pool
// can route its farthest LOD here with the same slot-ish calls.

import * as THREE from 'three';
import { OCT_GLSL, renderOctahedralViews, renderOctahedralCellRange } from './octahedral-impostor.js';

const _box = new THREE.Box3();
const _center = new THREE.Vector3();
const _size = new THREE.Vector3();

export class OctahedralImpostorTier {
  constructor(renderer, opts = {}) {
    this.renderer = renderer;
    this.grid = opts.grid ?? 8;
    this.cellPx = opts.cellPx ?? 64;             // 64px/view is ample for a sub-14px FAR LOD
    this.blend = opts.blend === true;            // bilinear cross-fade of nearest views
    // Capture the model at radius*padding so each octahedral cell carries a small
    // TRANSPARENT gutter. Cells are packed edge-to-edge in the atlas, so without
    // it a LinearFilter tap near a billboard edge bleeds the neighbouring view's
    // texels (faint cross-view ghosting); the gutter makes that bleed land on
    // alpha-0 texels instead. ~5% costs a hair of in-cell resolution.
    this.padding = opts.padding ?? 1.05;
    // maxLayers * atlasPx^2 * 4 bytes is the WHOLE array-texture VRAM, allocated
    // up front (WebGL2 texStorage3D has no per-layer growth). 128 * 512^2 * 4 =
    // ~134 MB; the old 256 * 1024^2 * 4 was ~1 GB and its one-shot allocation was
    // itself a ~120-240ms hitch on the first impostor (now eager-allocated at pool
    // construction, off the swap path). Raise maxLayers only with cellPx headroom.
    this.maxLayers = opts.maxLayers ?? 128;      // distinct assets with an impostor
    this.maxInstances = opts.maxInstances ?? 8192;
    this.atlasPx = this.grid * this.cellPx;

    // One WebGL2 array render target: layer L holds asset L's GRIDxGRID atlas.
    this.atlas = new THREE.WebGLArrayRenderTarget(this.atlasPx, this.atlasPx, this.maxLayers, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      generateMipmaps: false,
    });
    this.atlas.texture.colorSpace = THREE.SRGBColorSpace;

    // asset.url -> { layer, radius, center } ; and the layer allocator.
    this._assetLayers = new Map();
    this._nextLayer = 0;

    // Billboard geometry: a unit quad (corner in [-1,1], uv in [0,1]) + a
    // per-instance layer attribute. position is a placeholder (shader builds the
    // world quad from `corner`); kept so three is happy and bounds exist.
    const geo = new THREE.BufferGeometry();
    const corner = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
    const uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(18), 3));
    geo.setAttribute('corner', new THREE.BufferAttribute(corner, 2));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    this._layerAttr = new THREE.InstancedBufferAttribute(new Float32Array(this.maxInstances), 1);
    this._layerAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aLayer', this._layerAttr);

    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      defines: this.blend ? { IMPOSTOR_BLEND: '' } : {},
      uniforms: {
        uAtlas: { value: this.atlas.texture },
        uGrid: { value: this.grid },
      },
      transparent: false,
      alphaTest: this.blend ? 0.0 : 0.5,
      depthWrite: true,
      depthTest: true,
      side: THREE.DoubleSide,
      vertexShader: /* glsl */`
        ${OCT_GLSL}
        in vec2 corner;
        in float aLayer;
        uniform float uGrid;
        out vec2 vUv;
        out vec2 vCellBase;
        out vec2 vEnc;
        out float vLayer;
        void main() {
          vec3 center = instanceMatrix[3].xyz;
          float radius = length(instanceMatrix[0].xyz);
          vec3 viewDir = normalize(cameraPosition - center);
          vec2 enc = octEncode(viewDir);
          vEnc = enc;
          vec3 right, up;
          #ifdef IMPOSTOR_BLEND
            // Quad faces the CONTINUOUS view; the fragment bilinearly cross-fades
            // the nearest captured cells (no hard cell-snap, so no orientation pop).
            octFrame(viewDir, right, up);
            vCellBase = vec2(0.0);
          #else
            vec2 cell = clamp(floor(enc * uGrid), vec2(0.0), vec2(uGrid - 1.0));
            vCellBase = cell / uGrid;
            octFrame(octDecode((cell + 0.5) / uGrid), right, up);
          #endif
          vec3 worldPos = center + (corner.x * right + corner.y * up) * radius;
          vUv = uv;
          vLayer = aLayer;
          gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
        }
      `,
      fragmentShader: /* glsl */`
        precision highp float;
        precision highp sampler2DArray;
        uniform highp sampler2DArray uAtlas;
        uniform float uGrid;
        in vec2 vUv;
        in vec2 vCellBase;
        in vec2 vEnc;
        in float vLayer;
        out vec4 fragColor;
        void main() {
          #ifdef IMPOSTOR_BLEND
            // Alpha-weighted bilinear blend across the 4 nearest octahedral cells.
            // Weights are the bilinear fractions (sum to 1); rgb is normalised by
            // accumulated alpha so transparent neighbours don't darken silhouettes.
            vec2 g = vEnc * uGrid - 0.5;
            vec2 base = floor(g);
            vec2 f = fract(g);
            vec3 rgb = vec3(0.0);
            float aSum = 0.0;
            for (int j = 0; j < 2; j++) {
              for (int i = 0; i < 2; i++) {
                vec2 cell = clamp(base + vec2(float(i), float(j)), vec2(0.0), vec2(uGrid - 1.0));
                float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
                vec2 uvc = (cell + vUv) / uGrid;
                vec4 t = texture(uAtlas, vec3(uvc, vLayer));
                rgb += t.rgb * t.a * w;
                aSum += t.a * w;
              }
            }
            if (aSum < 0.5) discard;
            fragColor = vec4(rgb / max(aSum, 1e-4), 1.0);
          #else
            vec2 atlasUv = vCellBase + vUv / uGrid;
            vec4 c = texture(uAtlas, vec3(atlasUv, vLayer));
            if (c.a < 0.5) discard;
            fragColor = c;
          #endif
        }
      `,
    });
    this.material = material;

    this.mesh = new THREE.InstancedMesh(geo, material, this.maxInstances);
    this.mesh.frustumCulled = false; // pool gates impostor LOD by distance already
    this.mesh.count = 0;
    this.mesh.name = 'octahedral-impostor-tier';
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    this._instances = new Map();  // entity -> instanceId
    this._free = [];              // recycled instance ids
    this._highWater = 0;          // max id ever used + 1 (== mesh.count target)
    this._mat4 = new THREE.Matrix4();

    // ---- Incremental on-the-fly baking --------------------------------------
    // Persistent lit scratch scene + ortho camera reused across every asset bake,
    // so one asset's grid*grid views can be spread over several frames (a cell
    // budget) instead of stalling a single frame with all of them. The object is
    // reparented in for just the cells rendered this frame and restored before the
    // main render (the bake runs inside update(), pre-render, so no flicker).
    this._bakeScene = new THREE.Scene();
    this._bakeScene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const bk = new THREE.DirectionalLight(0xffffff, 1.4); bk.position.set(1, 2, 1.5);
    const bf = new THREE.DirectionalLight(0xffffff, 0.8); bf.position.set(-1.5, -0.5, -1);
    this._bakeScene.add(bk, bf);
    this._bakeCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 4);
    this._bakeJobs = new Map(); // asset.url -> { layer, cellsDone, center, radius }
    this._cellsRendered = 0;    // cumulative cells baked (witness: per-frame delta <= budget)
  }

  // Bake (or fetch) the impostor atlas layer for an asset's object3D. Returns
  // { layer, radius, center } or null if the object has empty bounds. Idempotent
  // per asset.url — the heavy GRIDxGRID render happens once.
  bakeAsset(asset, object3D) {
    const existing = this._assetLayers.get(asset.url);
    if (existing) return existing;
    if (this._nextLayer >= this.maxLayers) return null; // atlas full

    object3D.updateWorldMatrix(true, true);
    _box.setFromObject(object3D);
    if (_box.isEmpty()) return null;
    _box.getCenter(_center);
    _box.getSize(_size);
    const radius = 0.5 * _size.length() * this.padding;
    if (!(radius > 0)) return null;

    const layer = this._nextLayer++;
    renderOctahedralViews(this.renderer, object3D, {
      grid: this.grid, cellPx: this.cellPx, center: _center, radius, target: this.atlas, layer,
    });

    const desc = { layer, radius, center: _center.clone() };
    this._assetLayers.set(asset.url, desc);
    return desc;
  }

  hasAsset(asset) { return this._assetLayers.has(asset.url); }
  hasJob(asset) { return this._bakeJobs.has(asset.url); }
  layerFor(asset) { const d = this._assetLayers.get(asset.url); return d ? d.layer : -1; }

  // Incremental bake: render up to `cellBudget` more octahedral cells of this
  // asset's atlas this frame. `object3D` (e.g. a live entity root) is reparented
  // into the persistent bake scene with its transform neutralised to asset-local,
  // the cells are rendered, and it is restored to its original parent/transform
  // before this function returns (and thus before the frame's main render, so no
  // flicker). Returns the number of cells rendered. When the final cell lands the
  // asset is promoted to _assetLayers (hasAsset() flips true).
  bakeChunk(asset, object3D, cellBudget) {
    if (this._assetLayers.has(asset.url) || !object3D) return 0;
    const total = this.grid * this.grid;
    let job = this._bakeJobs.get(asset.url);
    if (!job) {
      if (this._nextLayer >= this.maxLayers) return 0; // atlas full
      job = { layer: this._nextLayer++, cellsDone: 0, center: new THREE.Vector3(), radius: 0 };
      this._bakeJobs.set(asset.url, job);
    }
    if (cellBudget <= 0 || job.cellsDone >= total) return 0;

    // Reparent + neutralise transform so the capture is ASSET-LOCAL (every
    // instance reuses one atlas regardless of its world scale/position).
    const prevParent = object3D.parent;
    const prevAuto = object3D.matrixAutoUpdate;
    const prevPos = object3D.position.clone();
    const prevQuat = object3D.quaternion.clone();
    const prevScale = object3D.scale.clone();
    const prevVisible = object3D.visible;
    const visSaves = [];
    object3D.traverse((o) => { if (o.isMesh) { visSaves.push(o); o._impSaveVis = o.visible; o.visible = true; } });
    object3D.visible = true;
    object3D.position.set(0, 0, 0); object3D.quaternion.identity(); object3D.scale.set(1, 1, 1);
    object3D.matrixAutoUpdate = false;
    this._bakeScene.add(object3D); // reparents in
    object3D.updateWorldMatrix(true, true);

    let take = 0, empty = false;
    if (job.cellsDone === 0) {
      _box.setFromObject(object3D);
      if (_box.isEmpty()) empty = true;
      else { _box.getCenter(job.center); _box.getSize(_size); job.radius = 0.5 * _size.length() * this.padding; }
    }
    if (!empty && job.radius > 0) {
      take = Math.min(cellBudget, total - job.cellsDone);
      renderOctahedralCellRange(this.renderer, this._bakeScene, this._bakeCam, {
        grid: this.grid, cellPx: this.cellPx, center: job.center, radius: job.radius,
        target: this.atlas, layer: job.layer,
        cellStart: job.cellsDone, cellCount: take, clearFirst: job.cellsDone === 0,
      });
      job.cellsDone += take;
      this._cellsRendered += take;
    }

    // Restore object to its live parent + transform/visibility.
    if (prevParent) prevParent.add(object3D); else this._bakeScene.remove(object3D);
    object3D.position.copy(prevPos); object3D.quaternion.copy(prevQuat); object3D.scale.copy(prevScale);
    object3D.matrixAutoUpdate = prevAuto; object3D.visible = prevVisible;
    for (const o of visSaves) { o.visible = o._impSaveVis; delete o._impSaveVis; }
    object3D.updateWorldMatrix(true, true);

    if (empty) { this._bakeJobs.delete(asset.url); return 0; }
    if (job.cellsDone >= total) {
      this._assetLayers.set(asset.url, { layer: job.layer, radius: job.radius, center: job.center.clone() });
      this._bakeJobs.delete(asset.url);
    }
    return take;
  }

  // Entity-facing: place/refresh an impostor instance for `entity` on atlas
  // `layer` at WORLD `center` with WORLD `radius` (caller composes both from the
  // asset-local descriptor and the entity transform). Returns the instanceId.
  acquire(entity, layer, cx, cy, cz, radius) {
    let id = this._instances.get(entity);
    if (id == null) {
      id = this._free.length ? this._free.pop() : this._highWater++;
      this._instances.set(entity, id);
    }
    this._layerAttr.array[id] = layer;
    this._layerAttr.needsUpdate = true;
    this._writeInstance(id, cx, cy, cz, radius);
    if (this._highWater > this.mesh.count) this.mesh.count = this._highWater;
    return id;
  }

  // Update only the world position/radius of an existing instance (cheap move).
  setCenter(id, x, y, z, radius) {
    if (id < 0) return;
    this._writeInstance(id, x, y, z, radius ?? 1);
  }

  _writeInstance(id, x, y, z, r) {
    this._mat4.makeScale(r, r, r);
    this._mat4.setPosition(x, y, z);
    this.mesh.setMatrixAt(id, this._mat4);
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  release(entity) {
    const id = this._instances.get(entity);
    if (id == null) return;
    this._instances.delete(entity);
    // Park the instance at a degenerate scale so it rasterizes nothing until the
    // id is recycled (InstancedMesh has no per-instance hide flag).
    this._mat4.makeScale(0, 0, 0);
    this.mesh.setMatrixAt(id, this._mat4);
    this.mesh.instanceMatrix.needsUpdate = true;
    this._free.push(id);
    // Compact: if the highest-index instance was freed, shrink the high-water
    // mark (and mesh.count) past any now-trailing free ids so the renderer stops
    // iterating parked instances each frame.
    if (id === this._highWater - 1) {
      const freeSet = new Set(this._free);
      while (this._highWater > 0 && freeSet.has(this._highWater - 1)) this._highWater--;
      this._free = this._free.filter((f) => f < this._highWater);
      this.mesh.count = this._highWater;
    }
  }

  instanceIdFor(entity) {
    const id = this._instances.get(entity);
    return id == null ? -1 : id;
  }

  dispose() {
    this.atlas.dispose();
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

export default { OctahedralImpostorTier };
