"use client";
import styles from "./page.module.css";
import { useCallback, useEffect, useState } from "react";
import React from "react";
import Webcam from "react-webcam";

interface AlbumResult {
  id: number;
}

export default function Home() {
  const [artist, setArtist] = useState<string>("");
  const [album, setAlbum] = useState<string>("");
  const [hasQuery, setHasQuery] = useState<boolean>(false);
  const [gotResult, setGotResult] = useState<boolean>(false);
  const [result, setResult] = useState<AlbumResult | null>(null);

  const [imageBase64, setImageBase64] = useState<string | null>(null);

  const webcamRef = React.useRef<Webcam>(null);
  const capture = React.useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setImageBase64(imageSrc);
    } else {
      setImageBase64(null);
    }
  }, [webcamRef]);

  const videoConstraints = {
    width: { min: 100 },
    height: { min: 100 },
    aspectRatio: 1,
    facingMode: "environment",
  };

  useEffect(() => {
    setHasQuery(album.length > 0 && artist.length > 0);
  }, [album, artist]);

  // useEffect(() => {
  //   fetch("./api/openai/image_detection?image=hiii")
  //     .then((res) => res.json())
  //     .then((json) => console.log(json));
  // }, []);

  const doSearch = useCallback(() => {
    fetch(`/api/deezer/album?name=${album}`)
      .then((res) => res.json())
      .then((json) => {
        const albumResults = json.data;
        if (albumResults.length <= 0) {
          setGotResult(false);
        } else {
          for (let i = 0; i < albumResults.length; i++) {
            if (
              albumResults[i].artist.name.toLowerCase() === artist.toLowerCase()
            ) {
              setGotResult(true);
              setResult(albumResults[i]);
            }
          }
          setResult(null);
        }
      });
  }, [album, artist]);

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

        {imageBase64 && (
          <div
            onClick={() => {
              navigator.clipboard.writeText(imageBase64);
            }}
            style={{
              maxWidth: 600,
              height: 200,
              overflow: "scroll",
              overflowWrap: "anywhere",
            }}>
            {imageBase64}
          </div>
        )}

        <input
          placeholder="Artist name"
          type="text"
          value={artist}
          onChange={(e) => setArtist(e.target.value)}
        />
        <input
          placeholder="Album name"
          type="text"
          value={album}
          onChange={(e) => setAlbum(e.target.value)}
        />
        <button onClick={doSearch} disabled={!hasQuery}>
          Search for {album} by {artist}
        </button>

        {gotResult && result && (
          <iframe
            title="deezer-widget"
            src={`https://widget.deezer.com/widget/dark/album/${result.id}`}
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
