export function readFramePrediction(data: unknown): { label: string; confidence: number | null } {
  if (typeof data !== "object" || data === null || !("label" in data) || typeof data.label !== "string") {
    throw new Error("Invalid prediction response");
  }
  const score = "confidence" in data ? data.confidence : null;
  const confidence = !["", "no match", "no confidence"].includes(data.label)
    && typeof score === "number" && Number.isFinite(score) && score >= 0 && score <= 1
    ? score : null;
  return { label: data.label, confidence };
}

export function formatModelConfidence(value: number | null): string {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
    ? `${(value * 100).toFixed(1)}%` : "Unavailable";
}
