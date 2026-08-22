import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  sourcemap: true,
  // `clean` is off: build:wasm copies the .wasm into dist before tsup runs.
  clean: false,
});
