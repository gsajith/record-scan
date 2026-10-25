import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabaseClient";

export async function POST(req: NextRequest) {
  // Parse multipart form data (the image file)
  const formData = await req.formData();
  const file = formData.get("file") as File | null;

  if (!file) {
    return new NextResponse("No file uploaded", { status: 400 });
  }

  // Give the file a unique name
  const fileExt = file.name.split(".").pop();
  const fileName = `${crypto.randomUUID()}.${fileExt}`;
  const filePath = `uploads/${fileName}`;

  // Upload to Supabase Storage
  const { error } = await supabase.storage
    .from("album-images")
    .upload(filePath, file, { contentType: file.type });

  if (error) {
    console.error(error);
    return new NextResponse("Upload failed", { status: 500 });
  }

  // Get public URL
  const { data } = supabase.storage.from("album-images").getPublicUrl(filePath);
  return NextResponse.json({ url: data.publicUrl });
}
