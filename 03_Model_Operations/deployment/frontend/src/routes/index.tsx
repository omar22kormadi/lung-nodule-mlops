import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, lazy, Suspense } from "react";
import axios from "axios";
import CTSliceViewer, { type Slice } from "@/components/CTSliceViewer";

const CT3DViewer = lazy(() => import("@/components/CT3DViewer"));

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Lung Nodule AI" },
      { name: "description", content: "DICOM upload, detection and classification for lung nodules." },
    ],
  }),
  component: Index,
});

const API_URL = (import.meta as any).env?.VITE_API_URL || "https://lung-backend.polytecsousse.dev";

const theme = {
  bg: "#0f1117",
  panel: "#1a1d27",
  border: "#2a2d3a",
  text: "#e8e9ed",
  muted: "#9ca3af",
  accent: "#6366f1",
  success: "#34d399",
  danger: "#f43f5e",
};

const styles: Record<string, any> = {
  page: { minHeight: "100vh", background: theme.bg, color: theme.text, fontFamily: "'Segoe UI', system-ui, sans-serif" },
  header: { maxWidth: 1100, margin: "0 auto", padding: "28px 20px 12px" },
  title: { fontSize: 26, fontWeight: 700, margin: 0 },
  subtitle: { color: theme.muted, marginTop: 8, fontSize: 14 },
  main: { maxWidth: 1100, margin: "0 auto", padding: "0 20px 40px" },
  tabs: { display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 },
  card: { background: theme.panel, border: `1px solid ${theme.border}`, borderRadius: 12, padding: 20, marginBottom: 16 },
  error: { background: "rgba(244,63,94,0.15)", border: `1px solid ${theme.danger}`, color: "#fecaca", padding: 12, borderRadius: 8, marginBottom: 16 },
  img: { maxWidth: "100%", borderRadius: 8, border: `1px solid ${theme.border}` },
  statGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 12, marginBottom: 16 },
  stat: { background: theme.bg, border: `1px solid ${theme.border}`, borderRadius: 8, padding: 14 },
  statLabel: { fontSize: 11, color: theme.muted, textTransform: "uppercase", letterSpacing: 0.5 },
  statValue: { fontSize: 20, fontWeight: 700, color: theme.text, marginTop: 4 },
};

const tabBtn = (active: boolean): React.CSSProperties => ({
  padding: "10px 18px",
  borderRadius: 8,
  border: `1px solid ${active ? theme.accent : theme.border}`,
  background: active ? theme.accent : theme.panel,
  color: active ? "#fff" : theme.muted,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 13,
});

const btn = (variant: "primary" | "success" = "primary", disabled = false): React.CSSProperties => ({
  padding: "10px 20px",
  borderRadius: 8,
  border: "none",
  fontWeight: 600,
  cursor: disabled ? "not-allowed" : "pointer",
  opacity: disabled ? 0.6 : 1,
  background: variant === "primary" ? theme.accent : "#059669",
  color: "#fff",
  fontSize: 14,
});

const badge = (ok: boolean): React.CSSProperties => ({
  display: "inline-block",
  padding: "4px 10px",
  borderRadius: 6,
  fontSize: 12,
  fontWeight: 700,
  background: ok ? "rgba(52,211,153,0.2)" : "rgba(244,63,94,0.2)",
  color: ok ? theme.success : theme.danger,
});

type ApiResult = {
  visualization?: string;
  prediction?: any;
  nodules?: any[];
  num_nodules?: number;
  filename?: string;
  slices?: Slice[];
  lung_mesh?: { vertices: number[]; faces: number[] };
};

function Index() {
  const [tab, setTab] = useState("dicom");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [dicomResult, setDicomResult] = useState<ApiResult | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [mounted, setMounted] = useState(false);
  const [meshLoading, setMeshLoading] = useState(false);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [meshData, setMeshData] = useState<{ lung_mesh?: any; nodules?: any[] } | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dicomFiles, setDicomFiles] = useState<File[]>([]);
  const [feedback, setFeedback] = useState<Record<number, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setMounted(true);
    axios.get(`${API_URL}/health`).then((r) => setHealth(r.data)).catch(() => setHealth(null));
  }, []);

  const run = async (fn: () => Promise<void>) => {
    setLoading(true);
    setError(null);
    try {
      await fn();
    } catch (err: any) {
      const d = err.response?.data?.detail;
      setError(typeof d === "string" ? d : JSON.stringify(d || err.message));
    } finally {
      setLoading(false);
    }
  };

  const fetchMesh = async (sessionId: string) => {
    setMeshLoading(true);
    setMeshError(null);
    setMeshData(null);
    try {
      const { data } = await axios.get(`${API_URL}/api/mesh/dicom`, {
        params: { session_id: sessionId },
      });
      setMeshData({ lung_mesh: data.lung_mesh, nodules: data.nodules });
    } catch (err: any) {
      const d = err.response?.data?.detail;
      setMeshError(typeof d === "string" ? d : err.message || "Failed to load 3D mesh");
    } finally {
      setMeshLoading(false);
    }
  };

  const runInference = () =>
    run(async () => {
      if (!files.length || files.length < 10) {
        setError("Select at least 10 .dcm files (slices around the nodule).");
        return;
      }
      const fd = new FormData();
      for (const f of files) fd.append("files", f);
      setMeshData(null);
      setMeshError(null);
      setFeedback({});
      setSaved(false);
      setSaveMsg(null);
      setDicomFiles(files);
      const { data } = await axios.post(`${API_URL}/api/predict/dicom`, fd);
      setDicomResult(data);
      setSessionId(data?.session_id ?? null);
      if (data?.session_id) {
        fetchMesh(data.session_id);
      }
    });

  const saveScan = async () => {
    if (!sessionId) return;
    const confirmed = (dicomResult?.nodules ?? [])
      .map((_: any, i: number) => i)
      .filter((i) => feedback[i] === true);
    if (confirmed.length === 0) {
      setSaveMsg("No confirmed predictions to save.");
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      const fd = new FormData();
      fd.append("session_id", sessionId);
      fd.append("confirmed_indices", JSON.stringify(confirmed));
      for (const f of dicomFiles) fd.append("dicom_files", f);
      await axios.post(`${API_URL}/api/save/labeled`, fd);
      setSaved(true);
      setSaveMsg("✅ Scan saved successfully! Thank you for helping improve the model.");
      // Fire-and-forget Azure sync — runs in background, user never waits for it
      if (sessionId) {
        axios.post(`${API_URL}/api/sync/azure/${sessionId}`).catch(() => {});
      }
    } catch (err: any) {
      setSaveMsg(err.response?.data?.detail || err.message || "Failed to save scan");
    } finally {
      setSaving(false);
    }
  };




  const loadClassificationSample = () =>
    run(async () => {
      const { data } = await axios.get(`${API_URL}/api/test/classification/sample`);
      setResult(data);
    });

  const loadDetectionSample = () =>
    run(async () => {
      const { data } = await axios.get(`${API_URL}/api/test/detection/sample`);
      setResult(data);
    });


  const mal = result?.prediction?.malignancy;
  const topNodule = result?.nodules?.[0];

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>Lung Nodule AI</h1>
        <p style={styles.subtitle}>
          YOLOv8 detection (LUNA16) · R2Plus1D classification (LIDC-IDRI) · API v2
        </p>
        {health && (
          <p style={{ fontSize: 12, color: theme.muted }}>
            {health.device} · classifier {health.classifier ? "✓" : "✗"} · yolo {health.yolo ? "✓" : "✗"}
          </p>
        )}
      </header>

      <main style={styles.main}>
        <div style={styles.tabs}>
          {[
            ["dicom", "DICOM upload"],
            ["cls-test", "Classification test"],
            ["det-test", "Detection test"],
            
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              style={tabBtn(tab === id)}
              onClick={() => {
                setTab(id);
                setError(null);
                if (id !== "dicom") {
                  setResult(null);
                }
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {tab === "dicom" && (
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Upload CT slices (.dcm only)</h3>
            <p style={{ color: theme.muted, fontSize: 14, marginBottom: 16 }}>
              Upload ≥10 DICOM slices from the same CT study, then run inference.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
              <label
                htmlFor="dcm-input"
                style={{
                  ...btn("primary", false),
                  background: theme.bg,
                  color: theme.text,
                  border: `1px solid ${theme.border}`,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                📁 Choose .dcm files
              </label>
              <input
                id="dcm-input"
                type="file"
                multiple
                accept=".dcm"
                style={{ display: "none" }}
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
              />
              <span
                style={{
                  fontSize: 13,
                  color: files.length >= 10 ? theme.success : theme.muted,
                  fontWeight: 500,
                }}
              >
                {files.length
                  ? `${files.length} file${files.length === 1 ? "" : "s"} selected${files.length < 10 ? ` (need ≥10)` : ""}`
                  : "No files selected"}
              </span>
              <button
                type="button"
                style={btn("primary", loading || files.length < 10)}
                disabled={loading || files.length < 10}
                onClick={runInference}
              >
                {loading ? "Running inference…" : "Run inference"}
              </button>
            </div>
          </div>
        )}


        {tab === "cls-test" && (
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Classification test set</h3>
            <button type="button" style={btn("primary", loading)} disabled={loading} onClick={loadClassificationSample}>
              {loading ? "Loading…" : "Load random test sample"}
            </button>
          </div>
        )}

        {tab === "det-test" && (
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Detection test set</h3>
            <button type="button" style={btn("primary", loading)} disabled={loading} onClick={loadDetectionSample}>
              {loading ? "Loading…" : "Load random test slice"}
            </button>
          </div>
        )}


        {/* DICOM tab: only show results after inference completes */}
        {tab === "dicom" && dicomResult && (
          <>
            <div style={styles.card}>
              <h3 style={{ marginTop: 0 }}>Annotated CT slices</h3>
              <CTSliceViewer slices={dicomResult.slices ?? []} />

              {dicomResult.nodules?.length ? (
                <div style={{ marginTop: 16 }}>
                  {[...dicomResult.nodules]
                    .map((n: any, i: number) => ({ n, i }))
                    .sort((a, b) => (b.n.z ?? 0) - (a.n.z ?? 0))
                    .map(({ n, i }) => {
                      const totalSlices = dicomResult.slices?.length ?? 0;
                      const sliceNumber = totalSlices - (n.z ?? 0);
                      const ans = feedback[i];
                      return (
                        <div
                          key={i}
                          style={{
                            marginBottom: 8,
                            padding: "10px 12px",
                            background: theme.bg,
                            borderRadius: 8,
                            fontSize: 13,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            flexWrap: "wrap",
                            gap: 8,
                          }}
                        >
                          <div>
                            <div>
                              Slice {sliceNumber} · conf={n.confidence?.toFixed(3)}
                              {n.mal_prob != null && (
                                <span style={{ ...badge(n.mal_prob < 0.5), marginLeft: 10 }}>
                                  mal {(n.mal_prob * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                            {n.classification?.malignancy?.label && (
                              <div style={{ marginTop: 4, color: theme.muted }}>
                                {n.classification.malignancy.label}
                              </div>
                            )}
                          </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ color: theme.muted, fontSize: 12 }}>Correct?</span>
                          <button
                            type="button"
                            onClick={() => setFeedback((p) => ({ ...p, [i]: true }))}
                            style={{
                              ...btn("success", false),
                              padding: "5px 12px",
                              fontSize: 12,
                              opacity: ans === true ? 1 : ans === false ? 0.35 : 0.9,
                              outline: ans === true ? `2px solid ${theme.success}` : "none",
                            }}
                          >
                            ✅
                          </button>
                          <button
                            type="button"
                            onClick={() => setFeedback((p) => ({ ...p, [i]: false }))}
                            style={{
                              padding: "5px 12px",
                              fontSize: 12,
                              borderRadius: 8,
                              border: "none",
                              cursor: "pointer",
                              fontWeight: 600,
                              color: "#fff",
                              background: theme.danger,
                              opacity: ans === false ? 1 : ans === true ? 0.35 : 0.9,
                              outline: ans === false ? `2px solid ${theme.danger}` : "none",
                            }}
                          >
                            ❌
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </div>

            {dicomResult.nodules?.length ? (() => {
              const nodules = dicomResult.nodules!;
              const allAnswered = nodules.every((_: any, i: number) => feedback[i] !== undefined);
              const confirmedCount = nodules.filter((_: any, i: number) => feedback[i] === true).length;
              return (
                <>
                  {allAnswered && !saved && (
                    <div style={styles.card}>
                      <h3 style={{ marginTop: 0 }}>Save scan</h3>
                      <p style={{ margin: "0 0 12px", fontSize: 14 }}>
                        Would you like to save this scan to help improve the model?
                      </p>
                      {confirmedCount === 0 && (
                        <p style={{ margin: "0 0 12px", fontSize: 12, color: theme.muted }}>
                          Only confirmed nodules will be saved.
                        </p>
                      )}
                      {confirmedCount < nodules.length && confirmedCount > 0 && (
                        <p style={{ margin: "0 0 12px", fontSize: 12, color: theme.muted }}>
                          Only confirmed nodules will be saved.
                        </p>
                      )}
                      <button
                        type="button"
                        style={btn("success", saving)}
                        disabled={saving}
                        onClick={saveScan}
                      >
                        {saving ? "Saving…" : "Save Scan"}
                      </button>
                    </div>
                  )}
                  {saveMsg && (
                    <div style={styles.card}>
                      <div
                        style={{
                          padding: 12,
                          borderRadius: 8,
                          background: saved ? "rgba(52,211,153,0.15)" : "rgba(244,63,94,0.15)",
                          border: `1px solid ${saved ? theme.success : theme.danger}`,
                          color: saved ? theme.success : "#fecaca",
                          fontSize: 13,
                        }}
                      >
                        {saveMsg}
                      </div>
                    </div>
                  )}
                </>
              );
            })() : null}

            {mounted && (
              <Suspense
                fallback={
                  <div style={styles.card}>
                    <p style={{ color: theme.muted, fontSize: 13 }}>Loading 3D viewer…</p>
                  </div>
                }
              >
                <CT3DViewer
                  lungMesh={meshData?.lung_mesh}
                  nodules={(meshData?.nodules ?? dicomResult.nodules) as any}
                  loading={meshLoading}
                  error={meshError}
                />
              </Suspense>
            )}
          </>
        )}


        {/* Classification test: show predicted + true label with color coding */}
        {tab === "cls-test" && result?.visualization && (() => {
          const predicted = result.prediction?.malignancy?.label ?? null;
          const trueLabel =
            (result as any).true_label ??
            (result as any).ground_truth?.malignancy?.label ??
            (result as any).ground_truth?.label ??
            (result as any).label ??
            null;
          const correct =
            predicted && trueLabel
              ? String(predicted).toLowerCase() === String(trueLabel).toLowerCase()
              : null;
          const predColor = correct === null ? theme.text : correct ? theme.success : theme.danger;
          const prob = result.prediction?.malignancy?.probability_malignant;
          return (
            <div style={styles.card}>
              <h3 style={{ marginTop: 0 }}>Prediction result</h3>
              <p style={{ margin: "4px 0", fontSize: 15 }}>
                Predicted:{" "}
                <strong style={{ color: predColor }}>{predicted ?? "—"}</strong>
                {prob != null && (
                  <span style={{ color: theme.muted, marginLeft: 8, fontSize: 13 }}>
                    ({(prob * 100).toFixed(1)}% malignant)
                  </span>
                )}
              </p>
              <p style={{ margin: "4px 0 16px", fontSize: 15 }}>
                True label:{" "}
                <strong style={{ color: theme.text }}>{trueLabel ?? "—"}</strong>
                {correct !== null && (
                  <span
                    style={{
                      ...badge(correct),
                      marginLeft: 10,
                      background: correct ? "rgba(52,211,153,0.2)" : "rgba(244,63,94,0.2)",
                      color: correct ? theme.success : theme.danger,
                    }}
                  >
                    {correct ? "✓ correct" : "✗ wrong"}
                  </span>
                )}
              </p>
              <img src={`data:image/jpeg;base64,${result.visualization}`} alt="Result" style={styles.img} />
            </div>
          );
        })()}

        {/* Other non-DICOM tabs keep legacy visualization rendering */}
        {tab !== "dicom" && tab !== "cls-test" && result?.visualization && (
          <div style={styles.card}>
            <h3 style={{ marginTop: 0 }}>Visualization</h3>
            {result.prediction?.malignancy && (
              <p style={{ color: theme.muted }}>
                Predicted: <strong style={{ color: theme.text }}>{result.prediction.malignancy.label}</strong>{" "}
                ({(result.prediction.malignancy.probability_malignant * 100).toFixed(1)}% malignant)
              </p>
            )}
            <img src={`data:image/jpeg;base64,${result.visualization}`} alt="Result" style={styles.img} />
          </div>
        )}
      </main>
    </div>
  );
}