"use client";

import { useEffect, useState } from "react";

export default function UploadBox({ file, onFileChange, onSubmit }) {
  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return (
    <div className="panel">
      <h2 className="upload-title">Upload Image</h2>
      <input
        className="input-file"
        type="file"
        accept="image/*"
        onChange={(e) => onFileChange(e.target.files?.[0] || null)}
      />

      {previewUrl && <img className="preview" src={previewUrl} alt="Preview" />}

      <button className="predict-btn" onClick={onSubmit}>
        Predict
      </button>
    </div>
  );
}
