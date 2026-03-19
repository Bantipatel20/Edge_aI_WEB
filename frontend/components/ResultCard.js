export default function ResultCard({ result, error }) {
  return (
    <div className="panel">
      <h2 className="result-title">Prediction Result</h2>

      {!result && !error && (
        <p>Upload an image and click Predict to see the model output.</p>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <p className="result-label">{result.label}</p>
          <p className="confidence">
            Confidence: {(result.confidence * 100).toFixed(2)}%
          </p>
          <span
            className={`badge ${
              result.confidence >= 0.7 ? "good" : "bad"
            }`}
          >
            {result.confidence >= 0.7 ? "High confidence" : "Low confidence"}
          </span>
        </>
      )}
    </div>
  );
}
