import { apiClient } from './client'

export async function listConversations() {
  return apiClient('/conversations')
}

export async function getConversationMessages(convId) {
  return apiClient(`/conversations/${convId}/messages`)
}
