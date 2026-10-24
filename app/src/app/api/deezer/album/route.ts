import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const album = searchParams.get("name");

  console.log(album);

  const res = await fetch(`	https://api.deezer.com/search/album?q="${album}"`, {
    method: "GET",
  });
  const latest = await res.json();

  return NextResponse.json(latest);
}
