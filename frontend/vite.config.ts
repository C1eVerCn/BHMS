import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

function resolveProxyTarget(env: Record<string, string>) {
  const explicitTarget = env.VITE_DEV_PROXY_TARGET?.trim() || env.BHMS_DEV_PROXY_TARGET?.trim()
  if (explicitTarget) {
    return explicitTarget
  }

  const apiBaseUrl = env.VITE_API_BASE_URL?.trim()
  if (apiBaseUrl && /^https?:\/\//.test(apiBaseUrl)) {
    const url = new URL(apiBaseUrl)
    return `${url.protocol}//${url.host}`
  }

  return 'http://127.0.0.1:8000'
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = resolveProxyTarget(env)

  return {
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
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
