/**
 * Client-side renderer: the browser draws the match, the server only simulates.
 *
 * The MJPEG path rasterises on the server and streams pixels. That cost 20-27 ms
 * a frame almost independently of resolution -- the expense is scene
 * construction, not pixels -- so per-player cameras were unaffordable (four
 * measured 112 ms against a 50 ms budget at 20 Hz) and every viewer saw the
 * same picture.
 *
 * Here the server sends STATE and each browser draws its own view on its own
 * GPU. Per-player cameras become free, the frame rate is the client's, and the
 * server's render cost goes to zero.
 *
 * Two feeds:
 *   /scene   once   90 geoms as primitives (type, size, rgba) -- 13 kB
 *   /poses   stream length-prefixed float32 [tick, t, xpos(3N), xmat(9N)]
 *
 * MuJoCo geom types map onto three.js primitives directly, which is why this
 * ships shapes rather than a mesh export: the whole scene is 90 of them.
 *
 * Z IS UP. MuJoCo is z-up and three.js defaults to y-up; rather than rotate
 * every pose on arrival, the whole scene lives in MuJoCo's frame and the
 * camera's `up` is set to +z. One line instead of a transform on every object
 * every tick, and no chance of a half-applied convention.
 */
import * as THREE from "/static/vendor/three.module.js";

const T_PLANE = 0, T_SPHERE = 2, T_CAPSULE = 3, T_CYLINDER = 5, T_BOX = 6;

export class SceneView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x5b7fa6);
    this.camera = new THREE.PerspectiveCamera(55, 1.5, 0.05, 400);
    this.camera.up.set(0, 0, 1);                 // z-up, see the header note
    this.meshes = [];
    this.n = 0;
    this.follow = null;        // player index for the chase camera, or null
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x448844, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(8, -12, 20);
    this.scene.add(sun);
    this._t = new THREE.Vector3();
    this._m = new THREE.Matrix4();
  }

  /** Build the meshes once from /scene. */
  build(desc) {
    this.desc = desc;
    for (const m of this.meshes) this.scene.remove(m);
    this.meshes = [];
    this.n = desc.geoms.length;
    for (const g of desc.geoms) {
      const geo = this._geometry(g);
      const [r, gr, b, a] = g.rgba;
      const mat = new THREE.MeshLambertMaterial({
        color: new THREE.Color(r, gr, b),
        transparent: a < 1, opacity: a,
        // The pitch is ~13 COPLANAR planes at z = 0 (a ground slab, the field
        // sections, the markings). Without a depth bias they z-fight and which
        // one shows is arbitrary per pixel and per frame. Offsetting by draw
        // index makes later geoms win deterministically, which matches the
        // order MuJoCo layers them in.
        polygonOffset: g.type === T_PLANE,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -4 * (g.i + 1),
      });
      const mesh = geo ? new THREE.Mesh(geo, mat) : new THREE.Object3D();
      mesh.matrixAutoUpdate = false;             // we write the matrix directly
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
  }

  _geometry(g) {
    const s = g.size;
    switch (g.type) {
      case T_PLANE:
        // MuJoCo size 0 on a plane means "infinite"; the pitch uses that for
        // the ground. Give it something large but finite to draw.
        return new THREE.PlaneGeometry(2 * (s[0] || 60), 2 * (s[1] || 60));
      // (Textured planes: see `textured` in the scene description -- MuJoCo
      // paints the ground slab with a grass texture we do not ship, so its raw
      // rgba is a flat grey. It is pushed to the BOTTOM of the plane stack
      // rather than recoloured, so the green field sections above it show.)
      case T_SPHERE:
        return new THREE.SphereGeometry(s[0], 20, 14);
      case T_CAPSULE:
        // MuJoCo capsule: size = (radius, HALF length of the cylindrical part),
        // aligned with the geom's local +z. three.js CapsuleGeometry takes the
        // cylinder length (not half), and is +y aligned, so it needs rotating.
        {
          const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 14);
          geo.rotateX(Math.PI / 2);
          return geo;
        }
      case T_CYLINDER:
        {
          const geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 26);
          geo.rotateX(Math.PI / 2);
          return geo;
        }
      case T_BOX:
        return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      default:
        return null;
    }
  }

  /** Apply one pose frame: Float32Array [tick, t, xpos(3N), xmat(9N)]. */
  apply(f) {
    const N = this.n, P = 2, M = 2 + 3 * N;
    for (let i = 0; i < N; i++) {
      const mesh = this.meshes[i];
      const p = P + 3 * i, r = M + 9 * i;
      // MuJoCo xmat is ROW-major; three.js `set` also takes row-major
      // arguments, so the rows go in as rows. (Its internal storage is
      // column-major, which `set` handles -- passing the array straight to
      // `fromArray` instead would transpose every orientation in the scene.)
      mesh.matrix.set(
        f[r + 0], f[r + 1], f[r + 2], f[p + 0],
        f[r + 3], f[r + 4], f[r + 5], f[p + 1],
        f[r + 6], f[r + 7], f[r + 8], f[p + 2],
        0, 0, 0, 1);
    }
  }

  /** Chase camera, matching the server's: behind the player, fixed tilt. */
  aim(playerBodyPos) {
    const c = this.desc.chase;
    if (this.follow !== null && playerBodyPos) {
      const sgn = this.follow < 2 ? 1 : -1;
      this.camera.position.set(playerBodyPos.x,
                               playerBodyPos.y - sgn * c.back, c.up);
      this._t.set(playerBodyPos.x, playerBodyPos.y, 0.4);
      this.camera.lookAt(this._t);
      this.camera.fov = c.fovy;
    } else {
      this.camera.position.set(0, -34, 24);
      this.camera.lookAt(0, 0, 0);
      this.camera.fov = 45;
    }
    this.camera.updateProjectionMatrix();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  draw() { this.renderer.render(this.scene, this.camera); }

  /**
   * Where a click lands on the pitch, in WORLD coordinates.
   *
   * The whole point of rendering locally: the browser owns the camera, so it
   * can do this itself and send world xy. The server's uv->world path existed
   * only because the server owned the camera, and it needed the two to agree
   * about a projection -- which is exactly where the server-side chase camera
   * went wrong once already.
   */
  pickGround(nx, ny) {
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const hit = new THREE.Vector3();
    return ray.ray.intersectPlane(plane, hit) ? { x: hit.x, y: hit.y } : null;
  }
}

/** Read the length-prefixed float32 stream, calling `onFrame` per frame. */
export async function streamPoses(url, onFrame, signal) {
  const res = await fetch(url, { signal });
  const reader = res.body.getReader();
  let buf = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    const next = new Uint8Array(buf.length + value.length);
    next.set(buf); next.set(value, buf.length);
    buf = next;
    // A chunk boundary can split a frame anywhere, so drain only whole frames
    // and keep the remainder. Dropping the tail here would desync the stream
    // permanently rather than visibly.
    for (;;) {
      if (buf.length < 4) break;
      const n = new DataView(buf.buffer, buf.byteOffset, 4).getUint32(0, true);
      if (buf.length < 4 + n) break;
      const body = buf.slice(4, 4 + n);
      onFrame(new Float32Array(body.buffer, body.byteOffset, n / 4));
      buf = buf.slice(4 + n);
    }
  }
}
