package chat

import (
	"context"
	"io"
	"net/http"
	"strings"
	"time"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
	dsprotocol "DeepSeek_Web_To_API/internal/deepseek/protocol"
	openaifmt "DeepSeek_Web_To_API/internal/format/openai"
	"DeepSeek_Web_To_API/internal/httpapi/openai/shared"
	"DeepSeek_Web_To_API/internal/sse"
	streamengine "DeepSeek_Web_To_API/internal/stream"
)

type chatNonStreamResult struct {
	rawThinking           string
	rawText               string
	thinking              string
	toolDetectionThinking string
	text                  string
	contentFilter         bool
	detectedCalls         int
	body                  map[string]any
	finishReason          string
	responseMessageID     int
}

func (h *Handler) handleNonStreamWithRetry(w http.ResponseWriter, ctx context.Context, a *auth.RequestAuth, resp *http.Response, payload map[string]any, pow, completionID, model, finalPrompt string, refFileTokens int, thinkingEnabled, exposeReasoning, searchEnabled bool, toolNames []string, toolsRaw any, requireToolCall bool, historySession *chatHistorySession, activeSessionID *string) {
	attempts := 0
	currentResp := resp
	usagePrompt := finalPrompt
	accumulatedThinking := ""
	accumulatedRawThinking := ""
	accumulatedToolDetectionThinking := ""
	for {
		result, ok := h.collectChatNonStreamAttempt(w, currentResp, completionID, model, usagePrompt, refFileTokens, thinkingEnabled, exposeReasoning, searchEnabled, toolNames, toolsRaw)
		if !ok {
			return
		}
		accumulatedThinking += sse.TrimContinuationOverlap(accumulatedThinking, result.thinking)
		accumulatedRawThinking += sse.TrimContinuationOverlap(accumulatedRawThinking, result.rawThinking)
		accumulatedToolDetectionThinking += sse.TrimContinuationOverlap(accumulatedToolDetectionThinking, result.toolDetectionThinking)
		result.thinking = accumulatedThinking
		result.rawThinking = accumulatedRawThinking
		result.toolDetectionThinking = accumulatedToolDetectionThinking
		detected := detectAssistantToolCalls(result.rawText, result.text, result.rawThinking, result.toolDetectionThinking, toolNames)
		result.detectedCalls = len(detected.Calls)
		result.body = openaifmt.BuildChatCompletionWithToolCallsVisibility(completionID, model, usagePrompt, result.thinking, result.text, detected.Calls, toolsRaw, exposeReasoning)
		addRefFileTokensToUsage(result.body, refFileTokens)
		result.finishReason = chatFinishReason(result.body)
		if !shouldRetryChatNonStream(result, attempts) {
			h.Auth.StoreParentMessageID(a, result.responseMessageID)
			h.finishChatNonStreamResult(w, result, attempts, usagePrompt, refFileTokens, requireToolCall, historySession)
			return
		}

		attempts++
		config.Logger.Info("[openai_empty_retry] attempting synthetic retry", "surface", "chat.completions", "stream", false, "retry_attempt", attempts, "parent_message_id", result.responseMessageID)
		retryPayload := clonePayloadForEmptyOutputRetry(payload, result.responseMessageID)
		retryPow, prepared := h.prepareChatEmptyOutputRetry(ctx, a, payload, retryPayload, pow, attempts, false, historySession, activeSessionID)
		if !prepared {
			h.finishChatNonStreamResult(w, result, attempts, usagePrompt, refFileTokens, requireToolCall, historySession)
			return
		}
		nextResp, err := h.DS.CallCompletion(ctx, a, retryPayload, retryPow, 3)
		if err != nil {
			writeCompletionCallError(w, historySession, err, result.thinking, result.text)
			config.Logger.Warn("[openai_empty_retry] retry request failed", "surface", "chat.completions", "stream", false, "retry_attempt", attempts, "error", err)
			return
		}
		usagePrompt = usagePromptWithEmptyOutputRetry(usagePrompt, attempts)
		currentResp = nextResp
	}
}

func (h *Handler) collectChatNonStreamAttempt(w http.ResponseWriter, resp *http.Response, completionID, model, usagePrompt string, refFileTokens int, thinkingEnabled, exposeReasoning, searchEnabled bool, toolNames []string, toolsRaw any) (chatNonStreamResult, bool) {
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		body, _ := io.ReadAll(resp.Body)
		writeOpenAIError(w, resp.StatusCode, string(body))
		return chatNonStreamResult{}, false
	}
	result := sse.CollectStream(resp, thinkingEnabled, true)
	stripReferenceMarkers := h.compatStripReferenceMarkers()
	finalThinking := cleanVisibleOutput(result.Thinking, stripReferenceMarkers)
	finalText := cleanVisibleOutput(result.Text, stripReferenceMarkers)
	if searchEnabled {
		finalText = replaceCitationMarkersWithLinks(finalText, result.CitationLinks)
	}
	detected := detectAssistantToolCalls(result.Text, finalText, result.Thinking, result.ToolDetectionThinking, toolNames)
	respBody := openaifmt.BuildChatCompletionWithToolCallsVisibility(completionID, model, usagePrompt, finalThinking, finalText, detected.Calls, toolsRaw, exposeReasoning)
	addRefFileTokensToUsage(respBody, refFileTokens)
	return chatNonStreamResult{
		rawThinking:           result.Thinking,
		rawText:               result.Text,
		thinking:              finalThinking,
		toolDetectionThinking: result.ToolDetectionThinking,
		text:                  finalText,
		contentFilter:         result.ContentFilter,
		detectedCalls:         len(detected.Calls),
		body:                  respBody,
		finishReason:          chatFinishReason(respBody),
		responseMessageID:     result.ResponseMessageID,
	}, true
}

func (h *Handler) finishChatNonStreamResult(w http.ResponseWriter, result chatNonStreamResult, attempts int, usagePrompt string, refFileTokens int, requireToolCall bool, historySession *chatHistorySession) {
	if requireToolCall && result.detectedCalls == 0 {
		message := "tool_choice requires at least one valid tool call."
		if historySession != nil {
			historySession.error(http.StatusUnprocessableEntity, message, "tool_choice_violation", result.thinking, result.text)
		}
		writeOpenAIErrorWithCode(w, http.StatusUnprocessableEntity, message, "tool_choice_violation")
		return
	}
	if result.detectedCalls == 0 && shouldWriteUpstreamEmptyOutputError(result.text) {
		status, message, code := upstreamEmptyOutputDetail(result.contentFilter, result.text, result.thinking)
		if historySession != nil {
			historySession.error(status, message, code, result.thinking, result.text)
		}
		writeUpstreamEmptyOutputError(w, result.text, result.thinking, result.contentFilter)
		config.Logger.Info("[openai_empty_retry] terminal empty output", "surface", "chat.completions", "stream", false, "retry_attempts", attempts, "success_source", "none", "content_filter", result.contentFilter)
		return
	}
	if historySession != nil {
		historySession.success(http.StatusOK, result.thinking, result.text, result.finishReason, openaifmt.BuildChatUsageForModel("", usagePrompt, result.thinking, result.text, refFileTokens))
	}
	writeJSON(w, http.StatusOK, result.body)
	source := "first_attempt"
	if attempts > 0 {
		source = "synthetic_retry"
	}
	config.Logger.Info("[openai_empty_retry] completed", "surface", "chat.completions", "stream", false, "retry_attempts", attempts, "success_source", source)
}

func chatFinishReason(respBody map[string]any) string {
	if choices, ok := respBody["choices"].([]map[string]any); ok && len(choices) > 0 {
		if fr, _ := choices[0]["finish_reason"].(string); strings.TrimSpace(fr) != "" {
			return fr
		}
	}
	return "stop"
}

func shouldRetryChatNonStream(result chatNonStreamResult, attempts int) bool {
	return emptyOutputRetryEnabled() &&
		attempts < emptyOutputRetryMaxAttempts() &&
		!result.contentFilter &&
		result.detectedCalls == 0 &&
		strings.TrimSpace(result.text) == ""
}

func (h *Handler) handleStreamWithRetry(w http.ResponseWriter, r *http.Request, a *auth.RequestAuth, resp *http.Response, payload map[string]any, pow, completionID, model, finalPrompt string, refFileTokens int, thinkingEnabled, exposeReasoning, searchEnabled bool, toolNames []string, toolsRaw any, requireToolCall bool, historySession *chatHistorySession, activeSessionID *string) {
	streamRuntime, initialType, ok := h.prepareChatStreamRuntime(w, resp, completionID, model, finalPrompt, refFileTokens, thinkingEnabled, exposeReasoning, searchEnabled, toolNames, toolsRaw, requireToolCall, historySession)
	if !ok {
		return
	}
	attempts := 0
	currentResp := resp
	for {
		terminalWritten, retryable := h.consumeChatStreamAttempt(r, currentResp, streamRuntime, initialType, thinkingEnabled, historySession, attempts < emptyOutputRetryMaxAttempts())
		if terminalWritten {
			h.Auth.StoreParentMessageID(a, streamRuntime.responseMessageID)
			logChatStreamTerminal(streamRuntime, attempts)
			return
		}
		if !retryable || !emptyOutputRetryEnabled() || attempts >= emptyOutputRetryMaxAttempts() {
			h.Auth.StoreParentMessageID(a, streamRuntime.responseMessageID)
			streamRuntime.finalize("stop", false)
			recordChatStreamHistory(streamRuntime, historySession)
			config.Logger.Info("[openai_empty_retry] terminal empty output", "surface", "chat.completions", "stream", true, "retry_attempts", attempts, "success_source", "none")
			return
		}
		attempts++
		config.Logger.Info("[openai_empty_retry] attempting synthetic retry", "surface", "chat.completions", "stream", true, "retry_attempt", attempts, "parent_message_id", streamRuntime.responseMessageID)
		retryPayload := clonePayloadForEmptyOutputRetry(payload, streamRuntime.responseMessageID)
		retryPow, prepared := h.prepareChatEmptyOutputRetry(r.Context(), a, payload, retryPayload, pow, attempts, true, historySession, activeSessionID)
		if !prepared {
			streamRuntime.finalize("stop", false)
			recordChatStreamHistory(streamRuntime, historySession)
			config.Logger.Info("[openai_empty_retry] terminal empty output", "surface", "chat.completions", "stream", true, "retry_attempts", attempts, "success_source", "none")
			return
		}
		nextResp, err := h.DS.CallCompletion(r.Context(), a, retryPayload, retryPow, 3)
		if err != nil {
			failChatStreamCompletionError(streamRuntime, historySession, err)
			config.Logger.Warn("[openai_empty_retry] retry request failed", "surface", "chat.completions", "stream", true, "retry_attempt", attempts, "error", err)
			return
		}
		if nextResp.StatusCode != http.StatusOK {
			defer func() { _ = nextResp.Body.Close() }()
			body, _ := io.ReadAll(nextResp.Body)
			failChatStreamRetry(streamRuntime, historySession, nextResp.StatusCode, string(body), "error")
			return
		}
		streamRuntime.finalPrompt = usagePromptWithEmptyOutputRetry(finalPrompt, attempts)
		currentResp = nextResp
	}
}

func (h *Handler) prepareChatEmptyOutputRetry(ctx context.Context, a *auth.RequestAuth, basePayload, retryPayload map[string]any, originalPow string, retryAttempt int, stream bool, historySession *chatHistorySession, activeSessionID *string) (string, bool) {
	var bindAuth func(*auth.RequestAuth)
	if historySession != nil {
		bindAuth = historySession.bindAuth
	}
	return shared.PrepareEmptyOutputRetry(ctx, h.Auth, h.DS, a, basePayload, retryPayload, originalPow, "chat.completions", stream, retryAttempt, bindAuth, activeSessionID)
}

func (h *Handler) prepareChatStreamRuntime(w http.ResponseWriter, resp *http.Response, completionID, model, finalPrompt string, refFileTokens int, thinkingEnabled, exposeReasoning, searchEnabled bool, toolNames []string, toolsRaw any, requireToolCall bool, historySession *chatHistorySession) (*chatStreamRuntime, string, bool) {
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		body, _ := io.ReadAll(resp.Body)
		if historySession != nil {
			historySession.error(resp.StatusCode, string(body), "error", "", "")
		}
		writeOpenAIError(w, resp.StatusCode, string(body))
		return nil, "", false
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	rc := http.NewResponseController(w)
	_, canFlush := w.(http.Flusher)
	if !canFlush {
		config.Logger.Warn("[stream] response writer does not support flush; streaming may be buffered")
	}
	initialType := "text"
	if thinkingEnabled {
		initialType = "thinking"
	}
	streamRuntime := newChatStreamRuntime(
		w, rc, canFlush, completionID, time.Now().Unix(), model, finalPrompt,
		thinkingEnabled, exposeReasoning, searchEnabled, h.compatStripReferenceMarkers(), toolNames, toolsRaw,
		requireToolCall,
		len(toolNames) > 0, h.toolcallFeatureMatchEnabled() && h.toolcallEarlyEmitHighConfidence(),
	)
	streamRuntime.refFileTokens = refFileTokens
	return streamRuntime, initialType, true
}

func (h *Handler) consumeChatStreamAttempt(r *http.Request, resp *http.Response, streamRuntime *chatStreamRuntime, initialType string, thinkingEnabled bool, historySession *chatHistorySession, allowDeferEmpty bool) (bool, bool) {
	defer func() { _ = resp.Body.Close() }()
	finalReason := "stop"
	streamengine.ConsumeSSE(streamengine.ConsumeConfig{
		Context:             r.Context(),
		Body:                resp.Body,
		ThinkingEnabled:     thinkingEnabled,
		InitialType:         initialType,
		KeepAliveInterval:   time.Duration(dsprotocol.KeepAliveTimeout) * time.Second,
		IdleTimeout:         time.Duration(dsprotocol.StreamIdleTimeout) * time.Second,
		MaxKeepAliveNoInput: dsprotocol.MaxKeepaliveCount,
	}, streamengine.ConsumeHooks{
		OnKeepAlive: streamRuntime.sendKeepAlive,
		OnParsed: func(parsed sse.LineResult) streamengine.ParsedDecision {
			decision := streamRuntime.onParsed(parsed)
			if historySession != nil {
				historySession.progress(streamRuntime.thinking.String(), streamRuntime.text.String())
			}
			return decision
		},
		OnFinalize: func(reason streamengine.StopReason, _ error) {
			if string(reason) == "content_filter" {
				finalReason = "content_filter"
			}
		},
		OnContextDone: func() {
			streamRuntime.markContextCancelled()
			if historySession != nil {
				historySession.stopped(streamRuntime.thinking.String(), streamRuntime.text.String(), string(streamengine.StopReasonContextCancelled))
			}
		},
	})
	if streamRuntime.finalErrorCode == string(streamengine.StopReasonContextCancelled) {
		return true, false
	}
	terminalWritten := streamRuntime.finalize(finalReason, allowDeferEmpty && finalReason != "content_filter")
	if terminalWritten {
		recordChatStreamHistory(streamRuntime, historySession)
		return true, false
	}
	return false, true
}

func recordChatStreamHistory(streamRuntime *chatStreamRuntime, historySession *chatHistorySession) {
	if historySession == nil {
		return
	}
	if streamRuntime.finalErrorMessage != "" {
		historySession.error(streamRuntime.finalErrorStatus, streamRuntime.finalErrorMessage, streamRuntime.finalErrorCode, streamRuntime.thinking.String(), streamRuntime.text.String())
		return
	}
	historySession.success(http.StatusOK, streamRuntime.finalThinking, streamRuntime.finalText, streamRuntime.finalFinishReason, streamRuntime.finalUsage)
}

func failChatStreamRetry(streamRuntime *chatStreamRuntime, historySession *chatHistorySession, status int, message, code string) {
	streamRuntime.sendFailedChunk(status, message, code)
	if historySession != nil {
		historySession.error(status, message, code, streamRuntime.thinking.String(), streamRuntime.text.String())
	}
}

func logChatStreamTerminal(streamRuntime *chatStreamRuntime, attempts int) {
	source := "first_attempt"
	if attempts > 0 {
		source = "synthetic_retry"
	}
	if streamRuntime.finalErrorCode == string(streamengine.StopReasonContextCancelled) {
		config.Logger.Info("[openai_empty_retry] terminal cancelled", "surface", "chat.completions", "stream", true, "retry_attempts", attempts, "error_code", streamRuntime.finalErrorCode)
		return
	}
	if streamRuntime.finalErrorMessage != "" {
		config.Logger.Info("[openai_empty_retry] terminal empty output", "surface", "chat.completions", "stream", true, "retry_attempts", attempts, "success_source", "none", "error_code", streamRuntime.finalErrorCode)
		return
	}
	config.Logger.Info("[openai_empty_retry] completed", "surface", "chat.completions", "stream", true, "retry_attempts", attempts, "success_source", source)
}
