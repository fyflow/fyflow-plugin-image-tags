import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // The declarations test imports fyflow's runtime, slower to load than vitest's default allows on a cold cache.
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});
