import { useMutation } from '@tanstack/react-query'
import { scanContractRisk } from '../api/risk'

export function useRiskReview() {
  return useMutation({
    mutationFn: (documentId) => scanContractRisk(documentId),
  })
}
