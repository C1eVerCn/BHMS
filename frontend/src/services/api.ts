import axios, { type AxiosRequestConfig } from 'axios'

export interface ApiEnvelope<T> {
  success: boolean
  message: string
  data: T
  error_code?: string
}

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api/v1',
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
