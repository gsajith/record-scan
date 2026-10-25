"use client";
import { useState, useRef, useEffect } from "react";
import imageCompression from "browser-image-compression";

export default function AnnotatePage() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [box, setBox] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
  } | null>(null);
  const [drawing, setDrawing] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const img = imgRef.current;
    if (!canvas || !ctx || !img) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (box) {
      ctx.strokeStyle = "#ff3333";
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.setLineDash([]);
    }
  }, [box, imageUrl]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    const compressed = await imageCompression(f, {
      maxWidthOrHeight: 1024, // scale down
      maxSizeMB: 0.5, // ~500 KB target
      useWebWorker: true,
    });

    const formData = new FormData();
    formData.append("file", compressed);

    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    setImageUrl(data.url);
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    setBox({ x: e.clientX - rect.left, y: e.clientY - rect.top, w: 0, h: 0 });
    setDrawing(true);
  }
  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawing || !box) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x2 = e.clientX - rect.left;
    const y2 = e.clientY - rect.top;
    setBox({ ...box, w: x2 - box.x, h: y2 - box.y });
  }
  function handleMouseUp() {
    setDrawing(false);
  }

  function getTouchPos(e: React.TouchEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const touch = e.touches[0];
    return {
      x: touch.clientX - rect.left,
      y: touch.clientY - rect.top,
    };
  }

  function handleTouch(
    e: React.TouchEvent<HTMLCanvasElement>,
    phase: "start" | "move"
  ) {
    e.preventDefault(); // prevents scroll on drag
    const { x, y } = getTouchPos(e);

    if (phase === "start") {
      setBox({ x, y, w: 0, h: 0 });
      setDrawing(true);
    } else if (phase === "move" && drawing && box) {
      setBox({ ...box, w: x - box.x, h: y - box.y });
    }
  }

  function handleTouchEnd() {
    setDrawing(false);
  }

  async function handleSave() {
    if (!imageUrl || !imgRef.current) return;
    const img = imgRef.current;

    let payload;

    if (box) {
      const xc = (box.x + box.w / 2) / img.width;
      const yc = (box.y + box.h / 2) / img.height;
      const w = Math.abs(box.w) / img.width;
      const h = Math.abs(box.h) / img.height;
      payload = {
        x_center: xc,
        y_center: yc,
        width: w,
        height: h,
        class_id: 0,
      };
    } else {
      // no box drawn
      payload = {
        x_center: null,
        y_center: null,
        width: null,
        height: null,
        class_id: -1,
      };
    }

    await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: imageUrl, ...payload }),
    });

    alert(box ? "Annotation saved!" : "Negative sample saved!");
    setBox(null);
    setImageUrl(null);
  }

  return (
    <div
      style={{
        padding: "16px",
        fontFamily: "system-ui, sans-serif",
        backgroundColor: "#fafafa",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}>
      <h2 style={{ marginBottom: "1rem", color: "#222" }}>
        Annotate Album Cover
      </h2>

      {!imageUrl && (
        <label
          style={{
            display: "inline-block",
            background: "#0070f3",
            color: "white",
            padding: "14px 22px",
            borderRadius: "8px",
            fontSize: "1.1rem",
            fontWeight: 500,
            cursor: "pointer",
            marginTop: "2rem",
            userSelect: "none",
          }}>
          📷 Take or Upload Photo
          <input
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleUpload}
            style={{ display: "none" }}
          />
        </label>
      )}

      {imageUrl && (
        <div
          style={{
            position: "relative",
            display: "inline-block",
            marginTop: "1rem",
          }}>
          <img
            ref={imgRef}
            src={imageUrl}
            alt="album"
            style={{
              maxWidth: "90vw",
              maxHeight: "70vh",
              borderRadius: "12px",
              boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
              display: "block",
            }}
            onLoad={(e) => {
              const img = e.currentTarget;
              const canvas = canvasRef.current;
              if (canvas) {
                canvas.width = img.width;
                canvas.height = img.height;
              }
            }}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              cursor: "crosshair",
              touchAction: "none",
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onTouchStart={(e) => handleTouch(e, "start")}
            onTouchMove={(e) => handleTouch(e, "move")}
            onTouchEnd={() => handleTouchEnd()}
          />
        </div>
      )}

      {imageUrl && (
        <button
          onClick={handleSave}
          style={{
            marginTop: "1.5rem",
            background: "#00b37e",
            color: "white",
            fontSize: "1.1rem",
            border: "none",
            borderRadius: "10px",
            padding: "14px 32px",
            boxShadow: "0 3px 6px rgba(0,0,0,0.15)",
            cursor: "pointer",
            touchAction: "manipulation",
          }}>
          ✅ Save Annotation
        </button>
      )}
    </div>
  );
}
