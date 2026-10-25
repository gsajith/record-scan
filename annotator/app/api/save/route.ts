import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabaseClient";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { image_url, x_center, y_center, width, height, class_id } = body;

    if (!image_url) {
      return new NextResponse("Missing image_url", { status: 400 });
    }

    // allow null coordinates for negative samples
    const isNegative =
      x_center === null ||
      y_center === null ||
      width === null ||
      height === null ||
      class_id === -1;

    const { error } = await supabase.from("annotations").insert([
      {
        image_url,
        x_center: isNegative ? null : x_center,
        y_center: isNegative ? null : y_center,
        width: isNegative ? null : width,
        height: isNegative ? null : height,
        class_id: isNegative ? -1 : class_id ?? 0,
      },
    ]);

    if (error) {
      console.error(error);
      return new NextResponse(error.message, { status: 500 });
    }

    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error(err);
    return new NextResponse("Server error", { status: 500 });
  }
}
