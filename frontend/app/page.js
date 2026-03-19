"use client";

import { useMemo, useState } from "react";
import axios from "axios";

import UploadBox from "../components/UploadBox";
import ResultCard from "../components/ResultCard";
import Loader from "../components/Loader";

export default function Home() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const apiBaseUrl = useMemo(
    () =>
      process.env.NEXT_PUBLIC_API_URL ||
      "http://localhost:10000",
    []
  );

  const handleUpload = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    setError("");
    setResult(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(`${apiBaseUrl}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult({
        label: res.data.prediction,
        confidence: Number(res.data.confidence || 0),
      });
    } catch (err) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        "Prediction failed. Check backend URL and model weights.";
      setError(detail);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="kicker">Edge AI Classifier</p>
        <h1>Drug Image Classification</h1>
        <p className="subtitle">
          Upload an image, send it to your FastAPI inference service, and get a
          confidence-scored prediction.
        </p>
      </section>

      <section className="card-grid">
        <UploadBox file={file} onFileChange={setFile} onSubmit={handleUpload} />
        <div className="result-panel">
          {isLoading && <Loader />}
          {!isLoading && <ResultCard result={result} error={error} />}
        </div>
      </section>
    </main>
  );
}
