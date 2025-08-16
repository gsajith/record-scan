import { NextRequest, NextResponse } from "next/server";

import OpenAI from "openai";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const base64Image = searchParams.get("image");

  console.log("image:", base64Image);

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const response = await openai.responses.create({
    model: "gpt-5-mini",
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: "what's in this image?" },
          {
            type: "input_image",
            image_url: `data:image/jpeg;base64,${base64Image}`,
            detail: "high",
          },
        ],
      },
    ],
  });
  console.log(response);

  return NextResponse.json(response);
}
