import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchDocuments, uploadDocument, getJobStatus } from '../api/documents'

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  })
}

export function useUploadDocument() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ file, allowedRoles }) => uploadDocument(file, allowedRoles),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })
}

export function useIngestionJob(jobId, options = {}) {
  const queryClient = useQueryClient()

  return useQuery({
    queryKey: ['ingestionJob', jobId],
    queryFn: () => getJobStatus(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'READY' || status === 'FAILED') {
        if (status === 'READY') {
          queryClient.invalidateQueries({ queryKey: ['documents'] })
        }
        return false
      }
      return 800 // Poll every 800ms while processing
    },
    ...options,
  })
}
