import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { askQuestionSync } from '../api/chat'
import { listConversations, getConversationMessages } from '../api/conversations'

export function useAskQuestion() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (payload) => askQuestionSync(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
    },
  })
}

export function useConversations() {
  return useQuery({
    queryKey: ['conversations'],
    queryFn: listConversations,
    staleTime: 60000,
  })
}

export function useConversationMessages(convId) {
  return useQuery({
    queryKey: ['conversationMessages', convId],
    queryFn: () => getConversationMessages(convId),
    enabled: Boolean(convId),
  })
}
