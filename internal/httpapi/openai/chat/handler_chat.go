package chat

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
	dsprotocol "DeepSeek_Web_To_API/internal/deepseek/protocol"
	openaifmt "DeepSeek_Web_To_API/internal/format/openai"
	"DeepSeek_Web_To_API/internal/httpapi/openai/shared"
	"DeepSeek_Web_To_API/internal/httpapi/requestbody"
	"DeepSeek_Web_To_API/internal/promptcompat"
	"DeepSeek_Web_To_API/internal/sse"
	streamengine "DeepSeek_Web_To_API/internal/stream"
)

func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Read body first so we can compute a session-affinity key before
	// acquiring an account from the pool. Same Claude Code / Codex
	// conversation → same account, preserving upstream session context.
	r.Body = http.MaxBytesReader(w, r.Body, openAIGeneralMaxSize)
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "too large") {
			writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		if errors.Is(err, requestbody.ErrInvalidUTF8Body) {
			writeOpenAIError(w, http.StatusBadRequest, "invalid json: invalid utf-8 request body")
			return
		}
		writeOpenAIError(w, http.StatusBadRequest, "invalid body")
		return
	}
	var req map[string]any
	if err := json.Unmarshal(rawBody, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid json")
		return
	}

	callerAuth, err := h.Auth.DetermineCaller(r)
	if err != nil {
		writeOpenAIError(w, http.StatusUnauthorized, err.Error())
		return
	}
	historyStdReq, err := promptcompat.NormalizeOpenAIChatRequest(h.Store, req, requestTraceID(r))
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	historyStdReq = shared.ApplyThinkingInjection(h.Store, historyStdReq)
	historySession := startQueuedChatHistory(h.ChatHistory, r, callerAuth, historyStdReq)

	a, err := h.Auth.DetermineWithSession(r, rawBody)
	if err != nil {
		status := http.StatusUnauthorized
		detail := err.Error()
		if err == auth.ErrNoAccount {
			status = http.StatusTooManyRequests
		}
		if historySession != nil {
			historySession.error(status, detail, "error", "", "")
		}
		writeOpenAIError(w, status, detail)
		return
	}
	if historySession != nil {
		historySession.bindAuth(a)
	}
	var sessionID string
	defer func() {
		h.autoDeleteRemoteSession(r.Context(), a, sessionID)
		h.Auth.Release(a)
	}()

	r = r.WithContext(auth.WithAuth(r.Context(), a))
	if err := h.preprocessInlineFileInputs(r.Context(), a, req); err != nil {
		if historySession != nil {
			historySession.error(http.StatusBadRequest, err.Error(), "error", "", "")
		}
		writeOpenAIInlineFileError(w, err)
		return
	}

	var stdReq promptcompat.StandardRequest
	if a.DeepSeekSessionID != "" {
		sessionID = a.DeepSeekSessionID
		config.Logger.Debug("[session] reusing existing DeepSeek session", "session_id", sessionID, "account", a.AccountID)
		stdReq, err = promptcompat.NormalizeOpenAIChatRequestIncremental(h.Store, req, requestTraceID(r))
	} else {
		stdReq, err = promptcompat.NormalizeOpenAIChatRequest(h.Store, req, requestTraceID(r))
	}
	if err != nil {
		if historySession != nil {
			historySession.error(http.StatusBadRequest, err.Error(), "error", "", "")
		}
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	stdReq = shared.ApplyThinkingInjection(h.Store, stdReq)
	stdReq, err = h.applyCurrentInputFile(r.Context(), a, stdReq)
	if err != nil {
		status, message := mapCurrentInputFileError(err)
		if historySession != nil {
			historySession.error(status, message, "error", "", "")
		}
		writeOpenAIError(w, status, message)
		return
	}
	if historySession != nil {
		historySession.updateHistoryText(stdReq.HistoryText)
	}

	if sessionID == "" {
		sessionID, err = h.DS.CreateSession(r.Context(), a, 3)
		if err != nil {
			sessionDetail := shared.SessionErrorDetail(err)
			if sessionDetail.Stopped || sessionDetail.Status == http.StatusGatewayTimeout {
				writeSessionCallError(w, historySession, err)
				return
			}
			if a.UseConfigToken {
				if historySession != nil {
					historySession.error(http.StatusUnauthorized, "Account token is invalid. Please re-login the account in admin.", "error", "", "")
				}
				writeOpenAIError(w, http.StatusUnauthorized, "Account token is invalid. Please re-login the account in admin.")
			} else {
				a.MarkDirectTokenInvalid()
				if historySession != nil {
					historySession.error(http.StatusUnauthorized, "Invalid token. If this should be a DeepSeek_Web_To_API key, add it to config.keys first.", "error", "", "")
				}
				writeOpenAIError(w, http.StatusUnauthorized, "Invalid token. If this should be a DeepSeek_Web_To_API key, add it to config.keys first.")
			}
			return
		}
		if sessionID != "" && a.SessionKey != "" {
			h.Auth.BindDeepSeekSession(a, sessionID)
			config.Logger.Debug("[session] bound new DeepSeek session", "session_id", sessionID, "account", a.AccountID, "session_key", a.SessionKey)
		}
	}
	pow, err := h.DS.GetPow(r.Context(), a, 3)
	if err != nil {
		powDetail := shared.PowErrorDetail(err)
		if powDetail.Stopped || powDetail.Status == http.StatusGatewayTimeout {
			writePowCallError(w, historySession, err)
			return
		}
		if !a.UseConfigToken {
			a.MarkDirectTokenInvalid()
		}
		if historySession != nil {
			historySession.error(http.StatusUnauthorized, "Failed to get PoW (invalid token or unknown error).", "error", "", "")
		}
		writeOpenAIError(w, http.StatusUnauthorized, "Failed to get PoW (invalid token or unknown error).")
		return
	}
	payload := stdReq.CompletionPayload(sessionID, a.ParentMessageID)
	resp, err := h.DS.CallCompletion(r.Context(), a, payload, pow, 3)
	if nextSessionID := strings.TrimSpace(asString(payload["chat_session_id"])); nextSessionID != "" {
		sessionID = nextSessionID
	}
	if err != nil {
		if !a.UseConfigToken && shared.CompletionErrorDetail(err).Status == http.StatusUnauthorized {
			a.MarkDirectTokenInvalid()
		}
		writeCompletionCallError(w, historySession, err, "", "")
		return
	}
	refFileTokens := stdReq.RefFileTokens
	if stdReq.Stream {
		h.handleStreamWithRetry(w, r, a, resp, payload, pow, sessionID, stdReq.ResponseModel, stdReq.FinalPrompt, refFileTokens, stdReq.Thinking, stdReq.ExposeReasoning, stdReq.Search, stdReq.ToolNames, stdReq.ToolsRaw, stdReq.ToolChoice.IsRequired(), historySession, &sessionID)
		return
	}
	h.handleNonStreamWithRetry(w, r.Context(), a, resp, payload, pow, sessionID, stdReq.ResponseModel, stdReq.FinalPrompt, refFileTokens, stdReq.Thinking, stdReq.ExposeReasoning, stdReq.Search, stdReq.ToolNames, stdReq.ToolsRaw, stdReq.ToolChoice.IsRequired(), historySession, &sessionID)
}

func (h *Handler) handleNonStream(w http.ResponseWriter, resp *http.Response, completionID, model, finalPrompt string, refFileTokens int, thinkingEnabled, exposeReasoning, searchEnabled bool, toolNames []string, toolsRaw any, historySession *chatHistorySession) {
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		body, _ := io.ReadAll(resp.Body)
		if historySession != nil {
			historySession.error(resp.StatusCode, string(body), "error", "", "")
		}
		writeOpenAIError(w, resp.StatusCode, string(body))
		return
	}
	result := sse.CollectStream(resp, thinkingEnabled, true)

	stripReferenceMarkers := h.compatStripReferenceMarkers()
	finalThinking := cleanVisibleOutput(result.Thinking, stripReferenceMarkers)
	finalText := cleanVisibleOutput(result.Text, stripReferenceMarkers)
	if searchEnabled {
		finalText = replaceCitationMarkersWithLinks(finalText, result.CitationLinks)
	}
	detected := detectAssistantToolCalls(result.Text, finalText, result.Thinking, result.ToolDetectionThinking, toolNames)
	if shouldWriteUpstreamEmptyOutputError(finalText) && len(detected.Calls) == 0 {
		status, message, code := upstreamEmptyOutputDetail(result.ContentFilter, finalText, finalThinking)
		if historySession != nil {
			historySession.error(status, message, code, finalThinking, finalText)
		}
		writeUpstreamEmptyOutputError(w, finalText, finalThinking, result.ContentFilter)
		return
	}
	respBody := openaifmt.BuildChatCompletionWithToolCallsVisibility(completionID, model, finalPrompt, finalThinking, finalText, detected.Calls, toolsRaw, exposeReasoning)
	addRefFileTokensToUsage(respBody, refFileTokens)
	finishReason := "stop"
	if choices, ok := respBody["choices"].([]map[string]any); ok && len(choices) > 0 {
		if fr, _ := choices[0]["finish_reason"].(string); strings.TrimSpace(fr) != "" {
			finishReason = fr
		}
	}
	if historySession != nil {
		historySession.success(http.StatusOK, finalThinking, finalText, finishReason, openaifmt.BuildChatUsageForModel(model, finalPrompt, finalThinking, finalText, refFileTokens))
	}
	writeJSON(w, http.StatusOK, respBody)
}

func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request, resp *http.Response, completionID, model, finalPrompt string, refFileTokens int, thinkingEnabled, exposeReasoning, searchEnabled bool, toolNames []string, toolsRaw any, historySession *chatHistorySession) {
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if historySession != nil {
			historySession.error(resp.StatusCode, string(body), "error", "", "")
		}
		writeOpenAIError(w, resp.StatusCode, string(body))
		return
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

	created := time.Now().Unix()
	bufferToolContent := len(toolNames) > 0
	emitEarlyToolDeltas := h.toolcallFeatureMatchEnabled() && h.toolcallEarlyEmitHighConfidence()
	stripReferenceMarkers := h.compatStripReferenceMarkers()
	initialType := "text"
	if thinkingEnabled {
		initialType = "thinking"
	}

	streamRuntime := newChatStreamRuntime(
		w,
		rc,
		canFlush,
		completionID,
		created,
		model,
		finalPrompt,
		thinkingEnabled,
		exposeReasoning,
		searchEnabled,
		stripReferenceMarkers,
		toolNames,
		toolsRaw,
		false,
		bufferToolContent,
		emitEarlyToolDeltas,
	)
	streamRuntime.refFileTokens = refFileTokens

	streamengine.ConsumeSSE(streamengine.ConsumeConfig{
		Context:             r.Context(),
		Body:                resp.Body,
		ThinkingEnabled:     thinkingEnabled,
		InitialType:         initialType,
		KeepAliveInterval:   time.Duration(dsprotocol.KeepAliveTimeout) * time.Second,
		IdleTimeout:         time.Duration(dsprotocol.StreamIdleTimeout) * time.Second,
		MaxKeepAliveNoInput: dsprotocol.MaxKeepaliveCount,
	}, streamengine.ConsumeHooks{
		OnKeepAlive: func() {
			streamRuntime.sendKeepAlive()
		},
		OnParsed: func(parsed sse.LineResult) streamengine.ParsedDecision {
			decision := streamRuntime.onParsed(parsed)
			if historySession != nil {
				historySession.progress(streamRuntime.thinking.String(), streamRuntime.text.String())
			}
			return decision
		},
		OnFinalize: func(reason streamengine.StopReason, _ error) {
			if string(reason) == "content_filter" {
				streamRuntime.finalize("content_filter", false)
			} else {
				streamRuntime.finalize("stop", false)
			}
			if historySession == nil {
				return
			}
			if streamRuntime.finalErrorMessage != "" {
				historySession.error(streamRuntime.finalErrorStatus, streamRuntime.finalErrorMessage, streamRuntime.finalErrorCode, streamRuntime.thinking.String(), streamRuntime.text.String())
				return
			}
			historySession.success(http.StatusOK, streamRuntime.finalThinking, streamRuntime.finalText, streamRuntime.finalFinishReason, streamRuntime.finalUsage)
		},
		OnContextDone: func() {
			if historySession != nil {
				historySession.stopped(streamRuntime.thinking.String(), streamRuntime.text.String(), string(streamengine.StopReasonContextCancelled))
			}
		},
	})
}
