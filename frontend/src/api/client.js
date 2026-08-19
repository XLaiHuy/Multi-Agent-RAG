/**
 * Centralized API Client with JWT Bearer attachment, base URL resolution, and unified error handling.
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export function getAuthToken() {
  return localStorage.getItem('rag_token') || ''
}

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem('rag_token', token)
  } else {
    localStorage.removeItem('rag_token')
  }
}

export async function apiClient(endpoint, options = {}) {
  const token = getAuthToken()
  const headers = {
    ...(options.headers || {}),
  }

  // Only set Content-Type to JSON if body is not FormData
  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json'
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    // Dispatch auth expired event
    window.dispatchEvent(new CustomEvent('auth:unauthorized'))
  }

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`
    try {
      const errJson = await response.json()
      errorDetail = errJson.detail || errJson.message || errorDetail
    } catch {
      // ignore non-json error response
    }
    const error = new Error(errorDetail)
    error.status = response.status
    throw error
  }

  // Handle blob responses (e.g. file content)
  if (options.responseType === 'blob') {
    return response.blob()
  }

  return response.json()
}
