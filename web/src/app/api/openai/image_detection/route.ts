import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { z } from "zod";
import { zodTextFormat } from "openai/helpers/zod";

const AlbumDataExtraction = z.object({
  album_name: z.string(),
  artist_name: z.array(z.string()),
  error: z.number(),
});

export async function POST(request: NextRequest) {
  const data = await request.json();
  const base64Image = data.image;

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
              text: "what vinyl record is in this image? Set the 'error' field to the percentage you are uncertain about the match, with 0 completely certain, and 1 being completely uncertain or some other parsing error occured, such as no vinyl albu detected in the image. Match the album name as exactly as you can to existing music databases, don't overexplain or add to the names.",
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
    return NextResponse.json({ error: 1 });
  }
}
