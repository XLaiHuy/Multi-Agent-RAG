import { apiClient } from './client'

export async function compareContracts({ docAId, docBId, facets }) {
  return apiClient('/compare', {
    method: 'POST',
    body: JSON.stringify({
      doc_a_id: docAId,
      doc_b_id: docBId,
      facets: facets || [],
    }),
  })
}
