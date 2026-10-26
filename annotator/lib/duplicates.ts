// Perceptual fingerprints, for catching the same wild photo added twice in different encodings.
import sharp from "sharp";

// Fingerprints this many bits apart (of 64) or fewer count as the same picture
const NEAR_DUPLICATE_BITS = 6;

/** 64-bit difference hash as 16 hex digits: shrink to 9x8 gray, one bit per pixel darker than its right neighbor. */
export async function fingerprint(image: Buffer): Promise<string> {
  const px = await sharp(image).grayscale().resize(9, 8, { fit: "fill" }).raw().toBuffer();
  let hex = "";
  for (let y = 0; y < 8; y++) {
    for (let half = 0; half < 2; half++) {
      let nibble = 0;
      for (let x = half * 4; x < half * 4 + 4; x++) nibble = (nibble << 1) | (px[y * 9 + x] < px[y * 9 + x + 1] ? 1 : 0);
      hex += nibble.toString(16);
    }
  }
  return hex;
}

export function isNearDuplicate(a: string, b: string): boolean {
  let bits = 0;
  for (let i = 0; i < a.length; i++) {
    for (let d = parseInt(a[i], 16) ^ parseInt(b[i], 16); d; d >>= 1) bits += d & 1;
  }
  return bits <= NEAR_DUPLICATE_BITS;
}
