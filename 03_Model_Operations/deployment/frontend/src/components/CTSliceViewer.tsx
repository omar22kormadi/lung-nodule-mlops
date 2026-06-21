import { useEffect, useMemo, useRef, useState } from "react";

export type Slice = { index: number; image: string };

const theme = {
  bg: "#0f1117",
  panel: "#1a1d27",
  border: "#2a2d3a",
  text: "#e8e9ed",
  muted: "#9ca3af",
  accent: "#6366f1",
};

export default function CTSliceViewer({ slices: slicesProp }: { slices: Slice[] }) {
  const slices = useMemo(() => [...(slicesProp ?? [])].reverse(), [slicesProp]);
  const [idx, setIdx] = useState(0);
  const [zoom, setZoom] = useState(1);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIdx(0);
    setZoom(1);
  }, [slicesProp]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setIdx((p) => {
        const next = p + (e.deltaY > 0 ? 1 : -1);
        return Math.max(0, Math.min(slices.length - 1, next));
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [slices.length]);

  if (!slices?.length) {
    return (
      <div
        style={{
          background: theme.bg,
          border: `1px dashed ${theme.border}`,
          borderRadius: 8,
          padding: 40,
          textAlign: "center",
          color: theme.muted,
          fontSize: 13,
        }}
      >
        No slices returned by the inference API.
      </div>
    );
  }

  const current = slices[idx];

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 10,
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        <div style={{ fontSize: 13, color: theme.muted }}>
          Slice <strong style={{ color: theme.text }}>{idx + 1}</strong> / {slices.length}
          {current?.index != null && (
            <span style={{ marginLeft: 8, opacity: 0.7 }}>(z={idx + 1})</span>
          )}
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            style={btnStyle}
          >
            −
          </button>
          <span style={{ fontSize: 12, color: theme.muted, minWidth: 40, textAlign: "center" }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(4, z + 0.25))}
            style={btnStyle}
          >
            +
          </button>
          <button type="button" onClick={() => setZoom(1)} style={btnStyle}>
            Reset
          </button>
        </div>
      </div>

      <div
        ref={wrapRef}
        style={{
          background: "#000",
          border: `1px solid ${theme.border}`,
          borderRadius: 8,
          overflow: "auto",
          maxHeight: 700,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: 580,
        }}
      >
        <img
          src={`data:image/jpeg;base64,${current.image}`}
          alt={`Slice ${idx + 1}`}
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: "center center",
            transition: "transform 0.1s",
            imageRendering: "auto",
            filter: "contrast(1.1) brightness(1.05)",
            width: zoom === 1 ? "auto" : "auto",
            height: zoom === 1 ? "560px" : "560px",
            maxWidth: zoom === 1 ? "100%" : "none",
            objectFit: "contain",
          }}
        />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 12 }}>
        <button
          type="button"
          onClick={() => setIdx((p) => Math.max(0, p - 1))}
          disabled={idx === 0}
          style={{ ...btnStyle, opacity: idx === 0 ? 0.4 : 1 }}
        >
          ◀ Prev
        </button>
        <input
          type="range"
          min={0}
          max={slices.length - 1}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
          style={{ flex: 1, accentColor: theme.accent }}
        />
        <button
          type="button"
          onClick={() => setIdx((p) => Math.min(slices.length - 1, p + 1))}
          disabled={idx === slices.length - 1}
          style={{ ...btnStyle, opacity: idx === slices.length - 1 ? 0.4 : 1 }}
        >
          Next ▶
        </button>
      </div>
      <p style={{ fontSize: 11, color: theme.muted, marginTop: 8 }}>
        Tip: scroll inside the viewer to step through slices.
      </p>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: 6,
  border: `1px solid ${theme.border}`,
  background: theme.panel,
  color: theme.text,
  cursor: "pointer",
  fontSize: 13,
  fontWeight: 600,
};