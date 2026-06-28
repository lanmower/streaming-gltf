// octahedral-impostor.js — on-the-fly octahedral impostors for the FINAL LOD.
//
// Why: past the vertex-color BatchedMesh far tier, even ~100-400 triangles per
// model is wasted at sub-pixel scale. An octahedral impostor replaces the whole
// model with ONE camera-facing quad that samples a pre-rendered atlas of the
// model seen from a grid of directions. Cost at distance collapses to a single
// textured quad per entity (and, in the instanced tier, one draw for thousands).
//
// "On-the-fly": the atlas is rendered in-browser from the already-loaded
// geometry the first time an asset needs the impostor LOD — no bake-time step,
// no extra download. bakeOctahedralImpostor() does GRIDxGRID render-to-texture
// passes into one atlas; the result is reused by every entity of that asset.
//
// Encoding: full-sphere octahedral mapping (Y up). Cell (i,j) of the GRIDxGRID
// atlas stores the model orthographically captured looking along the direction
// octDecode((i+0.5, j+0.5)/GRID). At display time the billboard is oriented to
// face the *quantized* nearest capture direction and textured with that cell, so
// in-cell reconstruction is exact (the only error is direction quantization,
// which a 3-tap blend can soften — see SHADER below).
//
// This is decode-only on the GPU and self-contained: it needs `three` and a live
// WebGLRenderer (passed in), nothing else.

import * as THREE from 'three';

// ----- octahedral encode/decode (must match the GLSL below byte-for-byte) -----
// dir (unit, Y-up) -> uv in [0,1]^2
export function octEncode(out, x, y, z) {
  const s = 1 / (Math.abs(x) + Math.abs(y) + Math.abs(z));
  let ox = x * s, oz = z * s;
  if (y < 0) {
    const ax = ox, az = oz;
    ox = (1 - Math.abs(az)) * (ax >= 0 ? 1 : -1);
    oz = (1 - Math.abs(ax)) * (az >= 0 ? 1 : -1);
  }
  out[0] = ox * 0.5 + 0.5;
  out[1] = oz * 0.5 + 0.5;
  return out;
}

// uv in [0,1]^2 -> unit dir (Y-up). Mirrors octDecode() in the shader.
export function octDecode(out, u, v) {
  const fx = u * 2 - 1, fz = v * 2 - 1;
  let x = fx, z = fz;
  let yv = 1 - Math.abs(fx) - Math.abs(fz);
  const t = Math.max(-yv, 0);
  x += x >= 0 ? -t : t;
  z += z >= 0 ? -t : t;
  const len = Math.hypot(x, yv, z) || 1;
  out[0] = x / len; out[1] = yv / len; out[2] = z / len;
  return out;
}

// ----- baking ----------------------------------------------------------------

const _box = new THREE.Box3();
const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const _dir = [0, 0, 0];
const _eye = new THREE.Vector3();
const _up = new THREE.Vector3();

// Render GRIDxGRID octahedral views of `object3D` into `target` (a
// WebGLRenderTarget, or a WebGLArrayRenderTarget + integer `layer`). The caller
// supplies the precomputed `center` (THREE.Vector3) and `radius`. Reused by both
// the single-atlas baker and the texture-array impostor tier so the capture
// convention stays identical to the display shader's octFrame().
//
// Reparents object into a scratch lit scene and restores it; fully saves and
// restores renderer state (target, viewport, scissor, autoClear, clearAlpha) —
// per-cell setViewport would otherwise leave the renderer in a corner rect.
export function renderOctahedralViews(renderer, object3D, opts) {
  const { grid, cellPx, center, radius, target, layer = 0 } = opts;

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(1, 2, 1.5);
  const fill = new THREE.DirectionalLight(0xffffff, 0.8); fill.position.set(-1.5, -0.5, -1);
  scene.add(key, fill);

  const prevParent = object3D.parent;
  const prevMatrixAuto = object3D.matrixAutoUpdate;
  scene.add(object3D); // reparents (removes from prevParent)

  const cam = new THREE.OrthographicCamera(-radius, radius, radius, -radius, 0.001, radius * 4);
  renderOctahedralCellRange(renderer, scene, cam, { grid, cellPx, center, radius, target, layer });

  if (prevParent) prevParent.add(object3D); else scene.remove(object3D);
  object3D.matrixAutoUpdate = prevMatrixAuto;
}

// Render a CONTIGUOUS sub-range of the grid*grid octahedral cells (row-major
// k = j*grid + i) into `target`/`layer` using a caller-supplied `scene` (already
// holding the object + lights) and ortho `cam` (reframed per call to `radius`).
// `clearFirst` clears the whole atlas/layer once before the first chunk; later
// chunks accumulate. This is the resumable primitive the impostor tier uses to
// spread one atlas across several frames (cell budget) instead of stalling a
// single frame with all grid*grid renders. Renderer state is fully restored.
export function renderOctahedralCellRange(renderer, scene, cam, opts) {
  const { grid, cellPx, center, radius, target, layer = 0,
    cellStart = 0, cellCount = grid * grid, clearFirst = true } = opts;

  // Reframe the (possibly shared) ortho camera to this asset's bound sphere.
  cam.left = -radius; cam.right = radius; cam.top = radius; cam.bottom = -radius;
  cam.near = 0.001; cam.far = radius * 4; cam.updateProjectionMatrix();

  const prevTarget = renderer.getRenderTarget();
  const prevAutoClear = renderer.autoClear;
  const prevScissorTest = renderer.getScissorTest();
  const prevClearAlpha = renderer.getClearAlpha();
  const prevViewport = renderer.getViewport(new THREE.Vector4());
  const prevScissor = renderer.getScissor(new THREE.Vector4());

  renderer.setRenderTarget(target, layer); // layer ignored for plain RTs
  renderer.setClearColor(0x000000, 0);
  renderer.autoClear = false;
  renderer.setScissorTest(true);
  if (clearFirst) renderer.clear(true, true, true); // one transparent clear for the layer

  const camDist = radius * 2;
  const total = grid * grid;
  const end = Math.min(cellStart + cellCount, total);
  for (let k = cellStart; k < end; k++) {
    const i = k % grid, j = Math.floor(k / grid);
    octDecode(_dir, (i + 0.5) / grid, (j + 0.5) / grid);
    _eye.set(center.x + _dir[0] * camDist, center.y + _dir[1] * camDist, center.z + _dir[2] * camDist);
    // Up reference: world Y unless the view is near-polar, then Z. MUST match
    // the frame the display shader rebuilds (octFrame in OCT_GLSL).
    if (Math.abs(_dir[1]) > 0.999) _up.set(0, 0, 1); else _up.set(0, 1, 0);
    cam.position.copy(_eye);
    cam.up.copy(_up);
    cam.lookAt(center);
    cam.updateMatrixWorld(true);

    const x = i * cellPx, y = j * cellPx;
    renderer.setViewport(x, y, cellPx, cellPx);
    renderer.setScissor(x, y, cellPx, cellPx);
    renderer.render(scene, cam);
  }

  renderer.setViewport(prevViewport);
  renderer.setScissor(prevScissor);
  renderer.setScissorTest(prevScissorTest);
  renderer.setRenderTarget(prevTarget);
  renderer.autoClear = prevAutoClear;
  renderer.setClearAlpha(prevClearAlpha);
}

// Render `object3D` into a GRIDxGRID octahedral atlas and return a descriptor:
//   { atlas: THREE.Texture, grid, radius, center: THREE.Vector3 }
// The object is temporarily reparented into a scratch scene, rendered, and
// restored to its original parent/transform — the caller's scene graph is left
// untouched. Renderer state (target, viewport, autoClear, scissor) is restored.
export function bakeOctahedralImpostor(renderer, object3D, opts = {}) {
  const grid = opts.grid ?? 8;                 // directions per axis (GRID^2 views)
  const cellPx = opts.cellPx ?? 128;           // pixels per captured view
  const padding = opts.padding ?? 1.05;        // ortho half-extent scale (>=1): >1 leaves a transparent cell gutter to stop cross-view bleed
  const atlasPx = grid * cellPx;

  // World bound sphere of the object (in its own local frame, transform-agnostic
  // so every instance reuses one atlas). We bake in the object's current world
  // transform's *orientation-free* space by measuring its untransformed bounds.
  object3D.updateWorldMatrix(true, true);
  _box.setFromObject(object3D);
  if (_box.isEmpty()) return null;
  _box.getCenter(_center);
  _box.getSize(_size);
  const radius = 0.5 * _size.length() * padding;
  if (!(radius > 0)) return null;

  // Scratch render target (RGBA so the background stays transparent -> alpha clip).
  const rt = new THREE.WebGLRenderTarget(atlasPx, atlasPx, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: true,
    stencilBuffer: false,
    generateMipmaps: false,
  });
  rt.texture.colorSpace = THREE.SRGBColorSpace;

  renderOctahedralViews(renderer, object3D, { grid, cellPx, center: _center, radius, target: rt });

  const tex = rt.texture;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;

  return { atlas: tex, renderTarget: rt, grid, radius, center: _center.clone() };
}

// ----- display ---------------------------------------------------------------

export const OCT_GLSL = /* glsl */`
vec2 octEncode(vec3 n) {
  n /= (abs(n.x) + abs(n.y) + abs(n.z));
  vec2 oct = n.xz;
  if (n.y < 0.0) oct = (1.0 - abs(vec2(oct.y, oct.x))) * vec2(oct.x >= 0.0 ? 1.0 : -1.0, oct.y >= 0.0 ? 1.0 : -1.0);
  return oct * 0.5 + 0.5;
}
vec3 octDecode(vec2 f) {
  f = f * 2.0 - 1.0;
  vec3 n = vec3(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
  float t = max(-n.y, 0.0);
  n.x += n.x >= 0.0 ? -t : t;
  n.z += n.z >= 0.0 ? -t : t;
  return normalize(n);
}
// Build the capture frame for a (quantized) view direction d — identical to the
// JS baker: up = worldY (or worldZ near-polar), right = normalize(cross(up,d)),
// up' = cross(d,right). Returns right in .xyz0 and up in .xyz1 via out params.
void octFrame(vec3 d, out vec3 right, out vec3 up) {
  vec3 up0 = (abs(d.y) > 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
  right = normalize(cross(up0, d));
  up = cross(d, right);
}
`;

// ShaderMaterial for a unit quad billboard (corner attribute in [-1,1]^2, uv in
// [0,1]). `impostor` is the descriptor from bakeOctahedralImpostor. The quad is
// oriented to face the nearest captured direction and textured with that cell.
//
// opts.instanced: read per-instance center+radius from instanceMatrix instead of
// the uCenter/uRadius uniforms (used by the instanced impostor tier).
export function makeImpostorMaterial(impostor, opts = {}) {
  const instanced = opts.instanced === true;
  const blend = opts.blend === true;
  const alphaTest = blend ? 0.0 : (opts.alphaTest ?? 0.5);

  const uniforms = {
    uAtlas: { value: impostor.atlas },
    uGrid: { value: impostor.grid },
    uCenter: { value: impostor.center.clone() },
    uRadius: { value: impostor.radius },
  };

  const defines = {};
  if (instanced) defines.IMPOSTOR_INSTANCED = '';
  if (blend) defines.IMPOSTOR_BLEND = '';

  const mat = new THREE.ShaderMaterial({
    uniforms,
    transparent: false,
    alphaTest,
    depthWrite: true,
    depthTest: true,
    side: THREE.DoubleSide,
    defines,
    vertexShader: /* glsl */`
      ${OCT_GLSL}
      attribute vec2 corner;
      uniform vec3 uCenter;
      uniform float uRadius;
      uniform float uGrid;
      varying vec2 vUv;
      varying vec2 vCellBase;
      varying vec2 vEnc;
      void main() {
        #ifdef IMPOSTOR_INSTANCED
          // instanceMatrix: translation = world center, scale.x = radius.
          vec3 center = vec3(instanceMatrix[3].xyz);
          float radius = length(instanceMatrix[0].xyz);
        #else
          vec3 center = uCenter;
          float radius = uRadius;
        #endif
        // View direction from impostor center toward the camera, world space.
        vec3 viewDir = normalize(cameraPosition - center);
        vec2 enc = octEncode(viewDir);
        vEnc = enc;
        vec3 right, up;
        #ifdef IMPOSTOR_BLEND
          // Quad faces the continuous view; fragment cross-fades nearest cells.
          octFrame(viewDir, right, up);
          vCellBase = vec2(0.0);
        #else
          // Quantize to nearest atlas cell center.
          vec2 cell = clamp(floor(enc * uGrid), vec2(0.0), vec2(uGrid - 1.0));
          vCellBase = cell / uGrid;
          octFrame(octDecode((cell + 0.5) / uGrid), right, up);
        #endif
        // Quad corner -> world. corner in [-1,1]; uv in [0,1].
        vec3 worldPos = center + (corner.x * right + corner.y * up) * radius;
        vUv = uv;
        gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      precision highp float;
      uniform sampler2D uAtlas;
      uniform float uGrid;
      varying vec2 vUv;
      varying vec2 vCellBase;
      varying vec2 vEnc;
      void main() {
        #ifdef IMPOSTOR_BLEND
          // Alpha-weighted bilinear blend across the 4 nearest octahedral cells.
          vec2 g = vEnc * uGrid - 0.5;
          vec2 base = floor(g);
          vec2 f = fract(g);
          vec3 rgb = vec3(0.0);
          float aSum = 0.0;
          for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
              vec2 cell = clamp(base + vec2(float(i), float(j)), vec2(0.0), vec2(uGrid - 1.0));
              float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
              vec4 t = texture2D(uAtlas, (cell + vUv) / uGrid);
              rgb += t.rgb * t.a * w;
              aSum += t.a * w;
            }
          }
          if (aSum < 0.5) discard;
          gl_FragColor = vec4(rgb / max(aSum, 1e-4), 1.0);
        #else
          vec4 c = texture2D(uAtlas, vCellBase + vUv / uGrid);
          if (c.a < 0.01) discard;
          gl_FragColor = c;
        #endif
      }
    `,
  });
  return mat;
}

// Convenience: a single billboard Mesh for one impostor (non-instanced). Place
// it at the model's world position; it self-orients per frame in the shader.
export function makeImpostorMesh(impostor, opts = {}) {
  const geo = new THREE.BufferGeometry();
  // Two triangles, corner in [-1,1], uv in [0,1].
  const corner = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
  const uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(18), 3)); // placeholder; shader ignores
  geo.setAttribute('corner', new THREE.BufferAttribute(corner, 2));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  const mat = makeImpostorMaterial(impostor, opts);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.name = 'octahedral-impostor';
  return mesh;
}

export default { bakeOctahedralImpostor, makeImpostorMaterial, makeImpostorMesh, octEncode, octDecode };
