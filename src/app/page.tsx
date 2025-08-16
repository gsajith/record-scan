"use client";
import styles from "./page.module.css";
import { useCallback, useEffect, useState } from "react";
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
  const [error, setError] = useState<number>(0);
  const [gotResult, setGotResult] = useState<boolean>(false);
  const [result, setResult] = useState<AlbumResult | null>(null);

  const [parsingImage, setParsingImage] = useState<boolean>(false);
  const [fetchingAlbum, setFetchingAlbum] = useState<boolean>(false);

  const webcamRef = React.useRef<Webcam>(null);

  const doSearch = () => {
    setFetchingAlbum(true);
    fetch(`/api/deezer/album?name=${album}`)
      .then((res) => res.json())
      .then((json) => {
        const albumResults = json.data;
        console.log("albumResults", albumResults);
        if (albumResults.length <= 0) {
          setGotResult(false);
        } else {
          for (let i = 0; i < albumResults.length; i++) {
            console.log(
              albumResults[i].artist.name.toLowerCase(),
              artist.includes(albumResults[i].artist.name.toLowerCase())
            );
            if (artist.includes(albumResults[i].artist.name.toLowerCase())) {
              setGotResult(true);
              setResult(albumResults[i]);
              setFetchingAlbum(false);
              return;
            }
          }
          setResult(null);
          setFetchingAlbum(false);
        }
      });
  };

  useEffect(() => {
    console.log("do search?", album);
    if (album.length > 0) {
      console.log("Searching...");
      doSearch();
    }
  }, [album]);

  const capture = React.useCallback(() => {
    setParsingImage(true);
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
          setParsingImage(false);
          if (json.error >= 0.95) {
            setArtist([]);
            setAlbum("");
            setError(json.error);
          } else {
            setArtist(json.artist_name.map((a: string) => a.toLowerCase()));
            setAlbum(json.album_name);
            setError(json.error);
          }
        });
    }
  }, [webcamRef]);

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
          <button disabled={parsingImage} onClick={capture}>
            Capture photo
          </button>
        </div>

        {parsingImage && <div>Parsing image...</div>}
        {fetchingAlbum && <div>Fetching album...</div>}

        {album.length > 0 && error < 0.95 && (
          <div>
            The album is {album} by {artist.join(", ")}!
          </div>
        )}

        {error >= 0.95 && (
          <div style={{ color: "red" }}>
            No vinyl found in this image. Thanks for wasting my freakin&#39; GPT
            tokens...
          </div>
        )}

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
