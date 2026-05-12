import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (
            id.includes('/echarts/charts') ||
            id.includes('/echarts/components') ||
            id.includes('/echarts/features') ||
            id.includes('/echarts/renderers')
          ) {
            return 'charts-modules'
          }
          if (id.includes('zrender')) return 'charts-renderer'
          if (id.includes('/@ant-design/icons') || id.includes('/@ant-design/icons-svg')) return 'icons'
          if (id.includes('/react-dom/') || id.includes('/react/') || id.includes('/scheduler/')) return 'react'
          if (id.includes('react-router')) return 'router'
          if (id.includes('axios')) return 'network'
        },
      },
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
