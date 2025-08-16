import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { z } from "zod";
import { zodTextFormat } from "openai/helpers/zod";

const AlbumDataExtraction = z.object({
  album_name: z.string(),
  artist_name: z.array(z.string()),
  error: z.boolean(),
});

export async function POST(request: NextRequest) {
  const data = await request.json();
  const base64Image = data.image;

  console.log("hit here", data);

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  try {
    const response = await openai.responses.create({
      model: "gpt-5-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: "what vinyl record is in this image? Set error to true if you are not sure, or if there are any unknowns which prevent your answer form being a certainty.",
            },
            {
              type: "input_image",
              image_url: base64Image,
              detail: "high",
            },
          ],
        },
      ],
      text: {
        format: zodTextFormat(AlbumDataExtraction, "album_data_extraction"),
      },
    });
    console.log(response);

    if (
      response.status === "incomplete" &&
      response.incomplete_details?.reason === "max_output_tokens"
    ) {
      // Handle the case where the model did not return a complete response
      throw new Error("Incomplete response");
    }

    // TODO: Handle refusals
    // console.log(math_response);
    // if (math_response.type === "refusal") {
    //   // handle refusal
    //   console.log(math_response.refusal);
    // } else if (math_response.type === "output_text") {
    //   console.log(math_response.text);
    // } else {
    //   throw new Error("No response content");
    // }

    return NextResponse.json(JSON.parse(response.output_text));
  } catch (e) {
    return NextResponse.json({ error: true });
  }
}
