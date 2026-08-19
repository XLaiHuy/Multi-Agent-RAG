import { useMutation } from '@tanstack/react-query'
import { compareContracts } from '../api/compare'

export function useContractCompare() {
  return useMutation({
    mutationFn: (payload) => compareContracts(payload),
  })
}
