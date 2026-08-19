import { apiClient, API_BASE_URL, getAuthToken } from './client'

export async function askQuestionSync(payload) {
  return apiClient('/chat', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function createChatStream(payload, onEvent, onError, onComplete) {
  const token = getAuthToken()
  const controller = new AbortController()

  fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(payload),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Streaming failed: HTTP ${response.status}`)
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const block of lines) {
          if (!block.trim()) continue
          let eventType = 'message'
          let eventData = ''

          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6).trim()
            }
          }

          if (eventType === 'done' || eventData === '[DONE]') {
            if (onComplete) onComplete()
            return
          }

          try {
            const parsed = JSON.parse(eventData)
            if (onEvent) onEvent(eventType, parsed)
          } catch {
            if (onEvent) onEvent(eventType, { raw: eventData })
          }
        }
      }
      if (onComplete) onComplete()
    })
    .catch((err) => {
      if (err.name !== 'AbortError' && onError) {
        onError(err)
      }
    })

  return () => controller.abort()
}
