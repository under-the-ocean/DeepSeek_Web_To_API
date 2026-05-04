package responses

import (
	"DeepSeek_Web_To_API/internal/toolcall"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
	dsprotocol "DeepSeek_Web_To_API/internal/deepseek/protocol"
	openaifmt "DeepSeek_Web_To_API/internal/format/openai"
	"DeepSeek_Web_To_API/internal/httpapi/historycapture"
	"DeepSeek_Web_To_API/internal/httpapi/openai/shared"
	"DeepSeek_Web_To_API/internal/promptcompat"
	"DeepSeek_Web_To_API/internal/sse"
	streamengine "DeepSeek_Web_To_API/internal/stream"
)

func (h *Handler) GetResponseByID(w http.ResponseWriter, r *http.Request) {
	a, err := h.Auth.DetermineCaller(r)
	if err != nil {
		writeOpenAIError(w, http.StatusUnauthorized, err.Error())
		return
	}

	id := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if id == "" {
		writeOpenAIError(w, http.StatusBadRequest, "response_id is required.")
		return
	}
	owner := responseStoreOwner(a)
	if owner == "" {
		writeOpenAIError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	st := h.getResponseStore()
	item, ok := st.get(owner, id)
	if !ok {
		writeOpenAIError(w, http.StatusNotFound, "Response not found.")
		return
	}
	writeJSON(w, http.StatusOK, item)
}

func (h *Handler) Responses(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, openAIGeneralMaxSize)
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "too large") {
			writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request body too large")
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
	traceID := requestTraceID(r)
	historyStdReq, err := promptcompat.NormalizeOpenAIResponsesRequest(h.Store, req, traceID)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	historyStdReq = shared.ApplyThinkingInjection(h.Store, historyStdReq)
	historySession := historycapture.StartWithStatus(h.ChatHistory, r, callerAuth, historyStdReq, "queued")

	a, err := h.Auth.DetermineWithSession(r, rawBody)
	if err != nil {
		status := http.StatusUnauthorized
		detail := err.Error()
		if err == auth.ErrNoAccount {
			status = http.StatusTooManyRequests
		}
		if historySession != nil {
			historySession.Error(status, detail, "error", "", "")
		}
		writeOpenAIError(w, status, detail)
		return
	}
	if historySession != nil {
		historySession.BindAuth(a)
	}
	defer h.Auth.Release(a)
	r = r.WithContext(auth.WithAuth(r.Context(), a))
	owner := responseStoreOwner(a)
	if owner == "" {
		if historySession != nil {
			historySession.Error(http.StatusUnauthorized, "unauthorized", "error", "", "")
		}
		writeOpenAIError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	if err := h.preprocessInlineFileInputs(r.Context(), a, req); err != nil {
		if historySession != nil {
			historySession.Error(http.StatusBadRequest, err.Error(), "error", "", "")
		}
		writeOpenAIInlineFileError(w, err)
		return
	}
	stdReq, err := promptcompat.NormalizeOpenAIResponsesRequest(h.Store, req, traceID)
	if err != nil {
		if historySession != nil {
			historySession.Error(http.StatusBadRequest, err.Error(), "error", "", "")
		}
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	stdReq = shared.ApplyThinkingInjection(h.Store, stdReq)
	stdReq, err = h.applyCurrentInputFile(r.Context(), a, stdReq)
	if err != nil {
		status, message := mapCurrentInputFileError(err)
		if historySession != nil {
			historySession.Error(status, message, "error", "", "")
		}
		writeOpenAIError(w, status, message)
		return
	}

	sessionID, err := h.DS.CreateSession(r.Context(), a, 3)
	if err != nil {
		handleCreateSessionError(w, historySession, a, err)
		return
	}
	pow, err := h.DS.GetPow(r.Context(), a, 3)
	if err != nil {
		handlePowError(w, historySession, a, err)
		return
	}
	payload := stdReq.CompletionPayload(sessionID, a.ParentMessageID)
	resp, err := h.DS.CallCompletion(r.Context(), a, payload, pow, 3)
	if err != nil {
		if !a.UseConfigToken && shared.CompletionErrorDetail(err).Status == http.StatusUnauthorized {
			a.MarkDirectTokenInvalid()
		}
		writeCompletionCallError(w, historySession, err, "", "")
		return
	}

	responseID := "resp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	refFileTokens := stdReq.RefFileTokens
	if stdReq.Stream {
		h.handleResponsesStreamWithRetry(w, r, a, resp, payload, pow, owner, responseID, stdReq.ResponseModel, stdReq.FinalPrompt, refFileTokens, stdReq.Thinking, stdReq.Search, stdReq.ToolNames, stdReq.ToolsRaw, stdReq.ToolChoice, traceID, historySession)
		return
	}
	h.handleResponsesNonStreamWithRetry(w, r.Context(), a, resp, payload, pow, owner, responseID, stdReq.ResponseModel, stdReq.FinalPrompt, refFileTokens, stdReq.Thinking, stdReq.Search, stdReq.ToolNames, stdReq.ToolsRaw, stdReq.ToolChoice, traceID, historySession)
}

func (h *Handler) handleResponsesNonStream(w http.ResponseWriter, resp *http.Response, owner, responseID, model, finalPrompt string, refFileTokens int, thinkingEnabled, searchEnabled bool, toolNames []string, toolsRaw any, toolChoice promptcompat.ToolChoicePolicy, traceID string) {
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		writeOpenAIError(w, resp.StatusCode, strings.TrimSpace(string(body)))
		return
	}
	result := sse.CollectStream(resp, thinkingEnabled, true)
	stripReferenceMarkers := h.compatStripReferenceMarkers()
	sanitizedThinking := cleanVisibleOutput(result.Thinking, stripReferenceMarkers)
	sanitizedText := cleanVisibleOutput(result.Text, stripReferenceMarkers)
	if searchEnabled {
		sanitizedText = replaceCitationMarkersWithLinks(sanitizedText, result.CitationLinks)
	}
	textParsed := detectAssistantToolCalls(result.Text, sanitizedText, result.Thinking, result.ToolDetectionThinking, toolNames)
	if len(textParsed.Calls) == 0 && writeUpstreamEmptyOutputError(w, sanitizedText, sanitizedThinking, result.ContentFilter) {
		return
	}
	logResponsesToolPolicyRejection(traceID, toolChoice, textParsed, "text")

	callCount := len(textParsed.Calls)
	if toolChoice.IsRequired() && callCount == 0 {
		writeOpenAIErrorWithCode(w, http.StatusUnprocessableEntity, "tool_choice requires at least one valid tool call.", "tool_choice_violation")
		return
	}

	responseObj := openaifmt.BuildResponseObjectWithToolCalls(responseID, model, finalPrompt, sanitizedThinking, sanitizedText, textParsed.Calls, toolsRaw)
	addRefFileTokensToUsage(responseObj, refFileTokens)
	h.getResponseStore().put(owner, responseID, responseObj)
	writeJSON(w, http.StatusOK, responseObj)
}

func (h *Handler) handleResponsesStream(w http.ResponseWriter, r *http.Request, resp *http.Response, owner, responseID, model, finalPrompt string, refFileTokens int, thinkingEnabled, searchEnabled bool, toolNames []string, toolsRaw any, toolChoice promptcompat.ToolChoicePolicy, traceID string) {
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		writeOpenAIError(w, resp.StatusCode, strings.TrimSpace(string(body)))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	rc := http.NewResponseController(w)
	_, canFlush := w.(http.Flusher)

	initialType := "text"
	if thinkingEnabled {
		initialType = "thinking"
	}
	bufferToolContent := len(toolNames) > 0
	emitEarlyToolDeltas := h.toolcallFeatureMatchEnabled() && h.toolcallEarlyEmitHighConfidence()
	stripReferenceMarkers := h.compatStripReferenceMarkers()

	streamRuntime := newResponsesStreamRuntime(
		w,
		rc,
		canFlush,
		responseID,
		model,
		finalPrompt,
		thinkingEnabled,
		searchEnabled,
		stripReferenceMarkers,
		toolNames,
		toolsRaw,
		bufferToolContent,
		emitEarlyToolDeltas,
		toolChoice,
		traceID,
		func(obj map[string]any) {
			h.getResponseStore().put(owner, responseID, obj)
		},
	)
	streamRuntime.refFileTokens = refFileTokens
	streamRuntime.sendCreated()

	streamengine.ConsumeSSE(streamengine.ConsumeConfig{
		Context:             r.Context(),
		Body:                resp.Body,
		ThinkingEnabled:     thinkingEnabled,
		InitialType:         initialType,
		KeepAliveInterval:   time.Duration(dsprotocol.KeepAliveTimeout) * time.Second,
		IdleTimeout:         time.Duration(dsprotocol.StreamIdleTimeout) * time.Second,
		MaxKeepAliveNoInput: dsprotocol.MaxKeepaliveCount,
	}, streamengine.ConsumeHooks{
		OnParsed: streamRuntime.onParsed,
		OnFinalize: func(reason streamengine.StopReason, _ error) {
			if string(reason) == "content_filter" {
				streamRuntime.finalize("content_filter", false)
				return
			}
			streamRuntime.finalize("stop", false)
		},
	})
}

func logResponsesToolPolicyRejection(traceID string, policy promptcompat.ToolChoicePolicy, parsed toolcall.ToolCallParseResult, channel string) {
	rejected := filteredRejectedToolNamesForLog(parsed.RejectedToolNames)
	if !parsed.RejectedByPolicy || len(rejected) == 0 {
		return
	}
	config.Logger.Warn(
		"[responses] rejected tool calls by policy",
		"trace_id", strings.TrimSpace(traceID),
		"channel", channel,
		"tool_choice_mode", policy.Mode,
		"rejected_tool_names", strings.Join(rejected, ","),
	)
}

func filteredRejectedToolNamesForLog(names []string) []string {
	if len(names) == 0 {
		return nil
	}
	out := make([]string, 0, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		switch strings.ToLower(trimmed) {
		case "", "tool_name":
			continue
		default:
			out = append(out, trimmed)
		}
	}
	return out
}
