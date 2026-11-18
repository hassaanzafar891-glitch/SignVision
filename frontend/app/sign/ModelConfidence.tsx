import { formatModelConfidence } from "./prediction-confidence";

export default function ModelConfidence({ value }: { value: number | null }) {
  return (
    <div role="status" aria-live="polite" aria-atomic="true"
      title="Predicted-class softmax score, not a calibrated probability of correctness."
      style={{ position: "absolute", top: 58, left: 12, zIndex: 11, pointerEvents: "none",
        padding: "8px 12px", borderRadius: 10, background: "var(--surface)", color: "var(--ink)",
        border: "1px solid var(--line)", fontSize: 12 }}>
      <div>Model confidence <span style={{ color: "var(--ink-soft)" }}>(uncalibrated)</span></div>
      <strong>{formatModelConfidence(value)}</strong>
    </div>
  );
}
