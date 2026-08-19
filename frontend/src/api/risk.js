import { apiClient } from './client'

export async function scanContractRisk(documentId) {
  return apiClient(`/risk/scan/${documentId}`, {
    method: 'POST',
  })
}
