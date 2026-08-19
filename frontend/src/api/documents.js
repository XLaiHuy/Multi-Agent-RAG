import { apiClient, API_BASE_URL, getAuthToken } from './client'

export async function fetchDocuments() {
  return apiClient('/documents')
}

export async function getDocumentById(docId) {
  return apiClient(`/documents/${docId}`)
}

export async function uploadDocument(file, allowedRoles = 'admin,legal,finance,hr,user') {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('allowed_roles', allowedRoles)

  return apiClient('/documents/upload', {
    method: 'POST',
    body: formData,
  })
}

export async function getJobStatus(jobId) {
  return apiClient(`/documents/jobs/${jobId}`)
}

export async function fetchDocumentBlob(docId) {
  return apiClient(`/documents/${docId}/content`, {
    responseType: 'blob',
  })
}

export function getDocumentContentUrl(docId) {
  const token = getAuthToken()
  return `${API_BASE_URL}/documents/${docId}/content?token=${encodeURIComponent(token)}`
}
