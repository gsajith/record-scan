import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The repo root has its own lockfile (husky), so pin Turbopack to this app.
  turbopack: { root: __dirname },
};

export default nextConfig;
