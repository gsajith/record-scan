# Record Scan web prototype

The temporary web scanner. It opens the camera, sends a frame to GPT-5-mini to identify the
album, and plays a Deezer preview. The identifier plan's Task 10 points it at the local
retrieval service instead, and it retires once the native app exists.

Run it from `web/`: `npm install`, then `npm run dev`.

Env vars go in `web/.env`. `OPENAI_API_KEY` is required today. `RETRIEVAL_URL` joins it in
Task 10. The repo root `.env.example` lists both.

Vercel builds the app from this folder.
