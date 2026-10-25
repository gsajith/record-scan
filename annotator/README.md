# 🎨 Album Annotator

This app now lives in the record-scan monorepo and is a temporary tool for gathering detector training photos.

A lightweight **Next.js + Supabase** web app for capturing photos and annotating bounding boxes around **album covers**.Used to build a YOLOv8 training dataset for the Record Scan project.

## 🚀 Overview

The Album Annotator lets you:

- Take or upload a photo (works on mobile camera).

- Draw one bounding box around the album cover.

- Save normalized YOLO coordinates and image URL to Supabase.

- Run fully in the browser — mobile-friendly and Vercel-deployable.

## 🧱 Architecture

### Routes

- / → Redirects to /annotate

- /annotate → Main annotation UI

- /api/upload → Uploads an image to Supabase Storage and returns its public URL

- /api/save → Inserts a record into the annotations table

### Supabase Tables

**annotations**

- id (uuid)

- image_url (text)

- x_center (float)

- y_center (float)

- width (float)

- height (float)

- class_id (int, default 0)

- created_at (timestamp)

- optional: split (train / val)

### Supabase Storage

- Bucket: album-images

- Stores all uploaded photos.

- Publicly readable, with insert RLS policy enabled.

## ⚙️ Environment Variables

Create a .env.local file in this directory:

If deploying to Vercel, add the same values under **Settings → Environment Variables**.

## 🧑‍💻 Local Development

Install dependencies:

` npm install `

Run locally:

` npm run dev `

Then visit http://localhost:3000.

## 📱 Mobile Usage

- Uses capture="environment" so mobile browsers open the rear camera.

- Touch-friendly canvas for drawing a single bounding box.

- After saving, the UI resets for the next photo automatically.

## ☁️ Deployment (Vercel)

1.  Push this folder to GitHub.

2.  Go to [vercel.com/new](https://vercel.com/new) and import your repo.

3.  Add the two environment variables (NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY).

4.  Click **Deploy**.

The app will be live within a minute on a vercel.app domain.

## 🔗 Integration with the Dataset Pipeline

Images and labels saved through this app are exported into a YOLOv8 dataset using the pipeline's exporter, relative to this folder:

../pipeline/detector/export_yolo_dataset.py

That script connects to Supabase, downloads all annotations and images, splits them into train/val folders, and writes YOLO label files automatically.

## 🧠 Notes

- Only one bounding box per image (no multi-object support yet).

- Simple UI optimized for speed and consistency.

- Anonymous usage supported — no login required.

- Works offline once deployed (except for uploads).

## 🪪 License

MIT © Gautham Sajith
