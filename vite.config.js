import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    emptyOutDir: true,
    minify: "esbuild",
    outDir: "dist",
    rollupOptions: {
      input: [
        "index.html",
        "tools/siameseRecognizerLab.html",
        "tools/sigilSignDetectorLab.html",
        "tools/spellEffectLab.html",
        "tools/strokeTemplateMaker.html",
        "tools/strokeTemplateViewer.html"
      ]
    },
    sourcemap: false
  }
});
