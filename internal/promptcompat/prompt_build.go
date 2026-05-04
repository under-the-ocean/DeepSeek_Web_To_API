package promptcompat

import (
	"strings"

	"DeepSeek_Web_To_API/internal/prompt"
)

func buildOpenAIFinalPrompt(messagesRaw []any, toolsRaw any, traceID string, thinkingEnabled bool) (string, []string) {
	return BuildOpenAIPrompt(messagesRaw, toolsRaw, traceID, DefaultToolChoicePolicy(), thinkingEnabled)
}

func BuildOpenAIPrompt(messagesRaw []any, toolsRaw any, traceID string, toolPolicy ToolChoicePolicy, thinkingEnabled bool) (string, []string) {
	messages := NormalizeOpenAIMessagesForPrompt(messagesRaw, traceID)
	toolNames := []string{}
	if tools, ok := toolsRaw.([]any); ok && len(tools) > 0 {
		messages, toolNames = injectToolPrompt(messages, tools, toolPolicy)
	}
	return prompt.MessagesPrepareWithThinking(messages, thinkingEnabled), toolNames
}

// BuildOpenAIPromptForAdapter exposes the OpenAI-compatible prompt building flow so
// other protocol adapters (for example Gemini) can reuse the same tool/history
// normalization logic and remain behavior-compatible with chat/completions.
func BuildOpenAIPromptForAdapter(messagesRaw []any, toolsRaw any, traceID string, thinkingEnabled bool) (string, []string) {
	return buildOpenAIFinalPrompt(messagesRaw, toolsRaw, traceID, thinkingEnabled)
}

// BuildOpenAIPromptIncremental builds a prompt containing only the last user message
// and any pending tool results. This is used when reusing an existing DeepSeek session
// where conversation history is already maintained upstream.
func BuildOpenAIPromptIncremental(messagesRaw []any, toolsRaw any, traceID string, toolPolicy ToolChoicePolicy, thinkingEnabled bool) (string, []string) {
	incrementalMessages := extractIncrementalMessages(messagesRaw)
	if len(incrementalMessages) == 0 {
		return BuildOpenAIPrompt(messagesRaw, toolsRaw, traceID, toolPolicy, thinkingEnabled)
	}
	messages := NormalizeOpenAIMessagesForPrompt(incrementalMessages, traceID)
	toolNames := []string{}
	if tools, ok := toolsRaw.([]any); ok && len(tools) > 0 {
		messages, toolNames = injectToolPrompt(messages, tools, toolPolicy)
	}
	return prompt.MessagesPrepareWithThinking(messages, thinkingEnabled), toolNames
}

// extractIncrementalMessages extracts the last user message and any preceding
// tool/function messages that are part of the same turn. This is used for
// incremental prompts when reusing an existing DeepSeek session.
func extractIncrementalMessages(messagesRaw []any) []any {
	if len(messagesRaw) == 0 {
		return nil
	}
	lastUserIdx := -1
	for i := len(messagesRaw) - 1; i >= 0; i-- {
		msg, ok := messagesRaw[i].(map[string]any)
		if !ok {
			continue
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		if role == "user" {
			lastUserIdx = i
			break
		}
	}
	if lastUserIdx < 0 {
		return nil
	}
	startIdx := lastUserIdx
	for i := lastUserIdx - 1; i >= 0; i-- {
		msg, ok := messagesRaw[i].(map[string]any)
		if !ok {
			break
		}
		role := strings.ToLower(strings.TrimSpace(asString(msg["role"])))
		if role == "tool" || role == "function" {
			startIdx = i
			continue
		}
		break
	}
	return messagesRaw[startIdx:]
}
