import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

export type MeshData = { vertices: number[]; faces: number[] };
export type Nodule = {
  mal_prob?: number;
  confidence?: number;
  diameter_mm?: number;
  x?: number;
  y?: number;
  z?: number;
  mesh3D?: MeshData;
};

function buildGeometry(data?: MeshData): THREE.BufferGeometry | null {
  if (!data?.vertices?.length || !data?.faces?.length) return null;
  const geom = new THREE.BufferGeometry();
  const verts = new Float32Array(data.vertices);
  geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  const maxIdx = data.vertices.length / 3;
  const Indices = maxIdx > 65535 ? Uint32Array : Uint16Array;
  geom.setIndex(new THREE.BufferAttribute(new Indices(data.faces), 1));
  geom.computeVertexNormals();
  return geom;
}

function riskColor(p: number): string {
  return p >= 0.5 ? "#ff3b6b" : "#22c55e";
}

function LungMesh({ data }: { data?: MeshData }) {
  const geom = useMemo(() => buildGeometry(data), [data]);
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!geom) return null;
  return (
    <mesh geometry={geom}>
      <meshStandardMaterial
        color="#e0e0e0"
        transparent
        opacity={0.15}
        roughness={0.6}
        metalness={0.0}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function NoduleMesh({ nodule }: { nodule: Nodule }) {
  const geom = useMemo(() => buildGeometry(nodule.mesh3D), [nodule.mesh3D]);
  const color = riskColor(nodule.mal_prob ?? 0);
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const s = 1 + Math.sin(clock.elapsedTime * 2) * 0.04;
    ref.current.scale.setScalar(s);
  });
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!geom) return null;
  return (
    <mesh ref={ref} geometry={geom}>
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.5}
        roughness={0.3}
        metalness={0.1}
      />
    </mesh>
  );
}

function NoduleFallbackSphere({ nodule }: { nodule: Nodule }) {
  const color = riskColor(nodule.mal_prob ?? 0);
  const r = Math.max(2, (nodule.diameter_mm ?? 8) / 2);
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const s = 1 + Math.sin(clock.elapsedTime * 2) * 0.06;
    ref.current.scale.setScalar(s);
  });
  if (nodule.x == null || nodule.y == null || nodule.z == null) return null;
  return (
    <mesh ref={ref} position={[nodule.x, nodule.y, nodule.z]}>
      <sphereGeometry args={[r, 32, 32]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.6}
        roughness={0.3}
      />
    </mesh>
  );
}

function FitCamera({ lung, nodules }: { lung?: MeshData; nodules?: Nodule[] }) {
  const { camera } = useThree();
  useEffect(() => {
    const box = new THREE.Box3();
    const tmp = new THREE.Vector3();
    if (lung?.vertices?.length) {
      const v = lung.vertices;
      for (let i = 0; i < v.length; i += 3) {
        tmp.set(v[i], v[i + 1], v[i + 2]);
        box.expandByPoint(tmp);
      }
    }
    nodules?.forEach((n) => {
      if (n.x != null && n.y != null && n.z != null) {
        box.expandByPoint(new THREE.Vector3(n.x, n.y, n.z));
      }
    });
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 100);
    const persp = camera as THREE.PerspectiveCamera;
    const fov = (persp.fov * Math.PI) / 180;
    const dist = (maxDim / 2 / Math.tan(fov / 2)) * 2;
    camera.position.set(center.x + dist, center.y + dist * 0.4, center.z + dist);
    camera.lookAt(center);
    camera.near = dist / 200;
    camera.far = dist * 20;
    camera.updateProjectionMatrix();
  }, [lung, nodules, camera]);
  return null;
}

function Spinner({ label }: { label: string }) {
  return (
    <div
      style={{
        height: 520,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 14,
        color: "#9ca3af",
        fontSize: 13,
      }}
    >
      <div
        style={{
          width: 44,
          height: 44,
          border: "3px solid #2a2d3a",
          borderTopColor: "#6366f1",
          borderRadius: "50%",
          animation: "ctspin 0.9s linear infinite",
        }}
      />
      <div>{label}</div>
      <style>{`@keyframes ctspin { to { transform: rotate(360deg) } }`}</style>
    </div>
  );
}

export default function CT3DViewer({
  lungMesh,
  nodules,
  loading,
  error,
}: {
  lungMesh?: MeshData;
  nodules?: Nodule[];
  loading?: boolean;
  error?: string | null;
}) {
  const [open, setOpen] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);

  const target = useMemo<[number, number, number]>(() => {
    if (lungMesh?.vertices?.length) {
      const v = lungMesh.vertices;
      let sx = 0, sy = 0, sz = 0;
      const n = v.length / 3;
      for (let i = 0; i < v.length; i += 3) { sx += v[i]; sy += v[i + 1]; sz += v[i + 2]; }
      return [sx / n, sy / n, sz / n];
    }
    return [0, 0, 0];
  }, [lungMesh]);

  const hasData = !!(lungMesh?.vertices?.length || nodules?.some((n) => n.mesh3D?.vertices?.length || n.x != null));

  return (
    <div
      style={{
        background: "linear-gradient(180deg, #1a1d27 0%, #161924 100%)",
        border: "1px solid #2a2d3a",
        borderRadius: 12,
        padding: 20,
        marginBottom: 16,
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          background: "transparent",
          border: "none",
          color: "#e8e9ed",
          fontSize: 16,
          fontWeight: 700,
          textAlign: "left",
          cursor: "pointer",
          padding: 0,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>🫁 Full 3D CT Reconstruction</span>
        <span style={{ color: "#9ca3af", fontSize: 14 }}>{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div style={{ marginTop: 16 }}>
          {loading ? (
            <div style={{ background: "#000", borderRadius: 8, border: "1px solid #2a2d3a" }}>
              <Spinner label="Generating 3D Mesh… This may take a few seconds" />
            </div>
          ) : error ? (
            <div style={{ padding: 20, color: "#f43f5e", fontSize: 13 }}>{error}</div>
          ) : !hasData ? (
            <div
              style={{
                padding: 40,
                textAlign: "center",
                color: "#9ca3af",
                border: "1px dashed #2a2d3a",
                borderRadius: 8,
                fontSize: 13,
              }}
            >
              No 3D reconstruction data available.
            </div>
          ) : (
            <div
              style={{
                height: 540,
                background: "radial-gradient(circle at 50% 40%, #1c2238 0%, #05070d 80%)",
                borderRadius: 8,
                overflow: "hidden",
                border: "1px solid #2a2d3a",
                position: "relative",
              }}
            >
              <Canvas
                camera={{ position: [300, 200, 300], fov: 45 }}
                gl={{ antialias: true, alpha: false }}
                onCreated={({ gl }) => gl.setClearColor("#05070d")}
              >
                <ambientLight intensity={0.5} />
                <directionalLight position={[200, 300, 200]} intensity={1.0} />
                <directionalLight position={[-200, -100, -200]} intensity={0.4} color="#88aaff" />
                <LungMesh data={lungMesh} />
                {nodules?.map((n, i) =>
                  n.mesh3D?.vertices?.length ? (
                    <NoduleMesh key={i} nodule={n} />
                  ) : (
                    <NoduleFallbackSphere key={i} nodule={n} />
                  ),
                )}
                <FitCamera lung={lungMesh} nodules={nodules} />
                <OrbitControls
                  target={target}
                  enablePan
                  enableZoom
                  enableRotate
                  autoRotate={autoRotate}
                  autoRotateSpeed={0.6}
                />
              </Canvas>
              <button
                type="button"
                onClick={() => setAutoRotate((r) => !r)}
                style={{
                  position: "absolute",
                  top: 10,
                  left: 10,
                  background: "rgba(15,17,23,0.8)",
                  color: "#e8e9ed",
                  border: "1px solid #2a2d3a",
                  borderRadius: 6,
                  padding: "6px 10px",
                  fontSize: 12,
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                {autoRotate ? "⏸ Pause rotation" : "▶ Auto-rotate"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
