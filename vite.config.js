import { defineConfig } from 'vite'

const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = process.env.GITHUB_ACTIONS && repositoryName ? `/${repositoryName}/` : '/'

export default defineConfig({
  base,
  server: {
    port: 3000,
    open: true
  },
  build: {
    target: 'esnext',
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('@tensorflow-models/mobilenet')) return 'model-mobilenet'
          if (id.includes('@tensorflow-models/coco-ssd')) return 'model-coco-ssd'
          if (id.includes('@tensorflow')) return 'tensorflow'
        }
      }
    }
  }
})
