"use client";
import styles from "./page.module.css";
import { useCallback, useState } from "react";
import React from "react";
import Webcam from "react-webcam";

interface AlbumResult {
  id: number;
}

const videoConstraints = {
  width: { min: 100 },
  height: { min: 100 },
  aspectRatio: 1,
  facingMode: "environment",
};

export default function Home() {
  const [artist, setArtist] = useState<string[]>([]);
  const [album, setAlbum] = useState<string>("");
  const [error, setError] = useState<boolean>(false);
  const [gotResult, setGotResult] = useState<boolean>(false);
  const [result, setResult] = useState<AlbumResult | null>(null);

  const webcamRef = React.useRef<Webcam>(null);

  const doSearch = useCallback(() => {
    if (album.length > 0) {
      fetch(`/api/deezer/album?name=${album}`)
        .then((res) => res.json())
        .then((json) => {
          const albumResults = json.data;
          if (albumResults.length <= 0) {
            setGotResult(false);
          } else {
            for (let i = 0; i < albumResults.length; i++) {
              if (artist.includes(albumResults[i].artist.name.toLowerCase())) {
                setGotResult(true);
                setResult(albumResults[i]);
              }
            }
            setResult(null);
          }
        });
    }
  }, [album, artist]);

  const capture = React.useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      fetch(`./api/openai/image_detection`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageSrc }),
      })
        .then((res) => res.json())
        .then((json) => {
          if (json.error) {
            setArtist([]);
            setAlbum("");
            setError(true);
          } else {
            setArtist(json.artist_name.map((a: string) => a.toLowerCase()));
            setAlbum(json.album_name);
            setError(false);
            doSearch();
          }
        });
    }
  }, [webcamRef, doSearch]);

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <div style={{ maxWidth: "100%", border: "1px solid white" }}>
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={"100%"}
            height={"100%"}
            videoConstraints={videoConstraints}
          />
          <button onClick={capture}>Capture photo</button>
        </div>

        {album.length > 0 && !error && (
          <div>
            The album is {album} by {artist.join(", ")}!
          </div>
        )}

        {error && (
          <div style={{ color: "red" }}>
            No vinyl found in this image. Thanks for wasting my freakin&#39; GPT
            tokens...
          </div>
        )}

        {gotResult && result && (
          <iframe
            title="deezer-widget"
            src={`https://widget.deezer.com/widget/dark/album/150129642`}
            width="100%"
            height="300"
            style={{ borderRadius: 12 }}
            frameBorder="0"
            allowTransparency={true}
            allow="encrypted-media; clipboard-write"></iframe>
        )}
      </main>
    </div>
  );
}
