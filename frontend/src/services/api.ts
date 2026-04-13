import axios, { type AxiosRequestConfig } from 'axios'

export interface ApiEnvelope<T> {
  success: boolean
  message: string
  data: T
  error_code?: string
}

type ApiVersion = 'v1' | 'v2'

function stripTrailingSlash(value: string) {
  return value.replace(/\/+$/, '')
}

export function resolveApiBaseUrl(version: ApiVersion): string {
  const configured = stripTrailingSlash(import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1')

  if (configured.endsWith(`/api/${version}`)) {
    return configured
  }

  const otherVersion = version === 'v1' ? 'v2' : 'v1'
  if (configured.endsWith(`/api/${otherVersion}`)) {
    return `${configured.slice(0, -(`/api/${otherVersion}`).length)}/api/${version}`
  }

  if (configured.endsWith('/api')) {
    return `${configured}/${version}`
  }

  return `${configured}/api/${version}`
}

export const API_V1_BASE_URL = resolveApiBaseUrl('v1')
export const API_V2_BASE_URL = resolveApiBaseUrl('v2')

const api = axios.create({
  baseURL: API_V1_BASE_URL,
  timeout: 30000,
})

export async function request<T>(config: AxiosRequestConfig): Promise<T> {
  const response = await api.request<ApiEnvelope<T>>(config)
  const payload = response.data
  if (!payload.success) {
    throw new Error(payload.message || payload.error_code || '请求失败')
  }
  return payload.data
}

export default api
