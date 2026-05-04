package claude

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
	claudefmt "DeepSeek_Web_To_API/internal/format/claude"
	"DeepSeek_Web_To_API/internal/httpapi/historycapture"
	openaishared "DeepSeek_Web_To_API/internal/httpapi/openai/shared"
	"DeepSeek_Web_To_API/internal/prompt"
	"DeepSeek_Web_To_API/internal/promptcompat"
	"DeepSeek_Web_To_API/internal/sse"
	"DeepSeek_Web_To_API/internal/toolcall"
	"DeepSeek_Web_To_API/internal/util"
)

type claudeSessionAuthResolver interface {
	DetermineWithSession(req *http.Request, body []byte) (*auth.RequestAuth, error)
}

type claudeCallerAuthResolver interface {
	DetermineCaller(req *http.Request) (*auth.RequestAuth, error)
}

func (h *Handler) handleDirectClaudeIfAvailable(w http.ResponseWriter, r *http.Request, store ConfigReader) bool {
	if h == nil || h.Auth == nil || h.DS == nil {
		return false
	}

	r.Body = http.MaxBytesReader(w, r.Body, openaishared.GeneralMaxSize)
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "too large") {
			writeClaudeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return true
		}
		writeClaudeError(w, http.StatusBadRequest, "invalid body")
		return true
	}
	var req map[string]any
	if err := json.Unmarshal(raw, &req); err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid json")
		return true
	}
	norm, err := normalizeClaudeRequest(store, cloneMap(req))
	if err != nil {
		writeClaudeError(w, http.StatusBadRequest, err.Error())
		return true
	}
	exposeThinking := applyClaudeDirectThinkingPolicy(&norm, req)
	historySession := h.startQueuedDirectClaudeHistory(r, req, norm.Standard)

	a, err := h.determineDirectClaudeAuth(r, raw, req)
	if err != nil {
		if historySession != nil {
			historySession.Error(claudeAuthErrorStatus(err), err.Error(), "error", "", "")
		}
		writeClaudeError(w, claudeAuthErrorStatus(err), err.Error())
		return true
	}
	if historySession != nil {
		historySession.BindAuth(a)
	}
	defer h.Auth.Release(a)

	r = r.WithContext(auth.WithAuth(r.Context(), a))
	if historySession == nil {
		historySession = historycapture.Start(h.ChatHistory, r, a, norm.Standard)
	}
	sessionID, err := h.DS.CreateSession(r.Context(), a, 3)
	if err != nil {
		sessionDetail := openaishared.SessionErrorDetail(err)
		if sessionDetail.Stopped || sessionDetail.Status == http.StatusGatewayTimeout {
			writeClaudeSessionCallError(w, historySession, err)
			return true
		}
		if a.UseConfigToken {
			if historySession != nil {
				historySession.Error(http.StatusUnauthorized, "Account token is invalid. Please re-login the account in admin.", "error", "", "")
			}
			writeClaudeError(w, http.StatusUnauthorized, "Account token is invalid. Please re-login the account in admin.")
			return true
		}
		if historySession != nil {
			historySession.Error(http.StatusUnauthorized, "Invalid token. If this should be a DeepSeek_Web_To_API key, add it to config.keys first.", "error", "", "")
		}
		a.MarkDirectTokenInvalid()
		writeClaudeError(w, http.StatusUnauthorized, "Invalid token. If this should be a DeepSeek_Web_To_API key, add it to config.keys first.")
		return true
	}
	pow, err := h.DS.GetPow(r.Context(), a, 3)
	if err != nil {
		powDetail := openaishared.PowErrorDetail(err)
		if powDetail.Stopped || powDetail.Status == http.StatusGatewayTimeout {
			writeClaudePowCallError(w, historySession, err)
			return true
		}
		if historySession != nil {
			historySession.Error(http.StatusUnauthorized, "Failed to get PoW (invalid token or unknown error).", "error", "", "")
		}
		if !a.UseConfigToken {
			a.MarkDirectTokenInvalid()
		}
		writeClaudeError(w, http.StatusUnauthorized, "Failed to get PoW (invalid token or unknown error).")
		return true
	}
	payload := norm.Standard.CompletionPayload(sessionID, 0)
	resp, err := h.DS.CallCompletion(r.Context(), a, payload, pow, 3)
	if err != nil {
		config.Logger.Warn("[claude] completion request failed", "stream", norm.Standard.Stream, "error", err)
		if !a.UseConfigToken && openaishared.CompletionErrorDetail(err).Status == http.StatusUnauthorized {
			a.MarkDirectTokenInvalid()
		}
		writeClaudeCompletionCallError(w, historySession, err, "", "")
		return true
	}
	if norm.Standard.Stream {
		h.handleClaudeStreamRealtime(
			w,
			r,
			resp,
			norm.Standard.ResponseModel,
			norm.Standard.Messages,
			norm.Standard.Thinking,
			norm.Standard.Search,
			norm.Standard.ToolNames,
			norm.Standard.ToolsRaw,
			historySession,
		)
		return true
	}
	h.handleDirectClaudeNonStream(w, resp, norm, exposeThinking, historySession)
	return true
}

func (h *Handler) startQueuedDirectClaudeHistory(r *http.Request, req map[string]any, stdReq promptcompat.StandardRequest) *historycapture.Session {
	resolver, ok := h.Auth.(claudeCallerAuthResolver)
	if !ok {
		return nil
	}
	reqForCaller := r.Clone(r.Context())
	reqForCaller.Header = r.Header.Clone()
	reqForCaller.Header.Set(auth.SessionAffinityHeader, claudeSessionAffinityScope(r, req))
	callerAuth, err := resolver.DetermineCaller(reqForCaller)
	if err != nil {
		return nil
	}
	return historycapture.StartWithStatus(h.ChatHistory, r, callerAuth, stdReq, "queued")
}

func (h *Handler) determineDirectClaudeAuth(r *http.Request, raw []byte, req map[string]any) (*auth.RequestAuth, error) {
	reqForAuth := r.Clone(r.Context())
	reqForAuth.Header = r.Header.Clone()
	reqForAuth.Header.Set(auth.SessionAffinityHeader, claudeSessionAffinityScope(r, req))
	if resolver, ok := h.Auth.(claudeSessionAuthResolver); ok {
		return resolver.DetermineWithSession(reqForAuth, raw)
	}
	return h.Auth.Determine(reqForAuth)
}

func claudeAuthErrorStatus(err error) int {
	if errors.Is(err, auth.ErrNoAccount) {
		return http.StatusTooManyRequests
	}
	return http.StatusUnauthorized
}

func applyClaudeDirectThinkingPolicy(norm *claudeNormalizedRequest, original map[string]any) bool {
	if norm == nil {
		return false
	}
	enabled, hasOverride := util.ResolveThinkingOverride(original)
	if !hasOverride {
		enabled = !norm.Standard.Stream
	}
	if config.IsNoThinkingModel(norm.Standard.ResolvedModel) {
		enabled = false
		hasOverride = false
	}
	norm.Standard.Thinking = enabled
	norm.Standard.ExposeReasoning = hasOverride && enabled
	norm.Standard.FinalPrompt = prompt.MessagesPrepareWithThinking(toMessageMaps(norm.Standard.Messages), enabled)
	return norm.Standard.ExposeReasoning
}

func (h *Handler) handleDirectClaudeNonStream(w http.ResponseWriter, resp *http.Response, norm claudeNormalizedRequest, exposeThinking bool, historySessions ...*historycapture.Session) {
	historySession := firstHistorySession(historySessions)
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		body, _ := io.ReadAll(resp.Body)
		message := strings.TrimSpace(string(body))
		if historySession != nil {
			historySession.Error(resp.StatusCode, message, "error", "", "")
		}
		writeClaudeError(w, resp.StatusCode, message)
		return
	}

	result := sse.CollectStream(resp, norm.Standard.Thinking, true)
	stripReferenceMarkers := h.compatStripReferenceMarkers()
	finalThinking := cleanVisibleOutput(result.Thinking, stripReferenceMarkers)
	finalText := cleanVisibleOutput(result.Text, stripReferenceMarkers)
	if norm.Standard.Search {
		finalText = openaishared.ReplaceCitationMarkersWithLinks(finalText, result.CitationLinks)
	}
	if shouldWriteClaudeEmptyOutputError(finalText, finalThinking, result.ContentFilter, norm.Standard.ToolNames) {
		status, message, _ := openaishared.UpstreamEmptyOutputDetail(result.ContentFilter, finalText, finalThinking)
		if historySession != nil {
			historySession.Error(status, message, "empty_output", finalThinking, finalText)
		}
		writeClaudeError(w, status, message)
		return
	}

	body := claudefmt.BuildMessageResponse(
		fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		norm.Standard.ResponseModel,
		norm.Standard.Messages,
		finalThinking,
		finalText,
		norm.Standard.ToolNames,
		norm.Standard.ToolsRaw,
	)
	raw, err := json.Marshal(body)
	if err != nil {
		if historySession != nil {
			historySession.Error(http.StatusInternalServerError, "failed to encode response", "error", finalThinking, finalText)
		}
		writeClaudeError(w, http.StatusInternalServerError, "failed to encode response")
		return
	}
	if historySession != nil {
		historySession.Success(http.StatusOK, finalThinking, finalText, "end_turn", nil)
	}
	if !exposeThinking {
		raw = stripClaudeThinkingBlocks(raw)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(raw)
}

func firstHistorySession(historySessions []*historycapture.Session) *historycapture.Session {
	if len(historySessions) == 0 {
		return nil
	}
	return historySessions[0]
}

func shouldWriteClaudeEmptyOutputError(finalText, finalThinking string, contentFilter bool, toolNames []string) bool {
	if !openaishared.ShouldWriteUpstreamEmptyOutputError(finalText) {
		return false
	}
	if len(toolcall.ParseToolCalls(finalText, toolNames)) > 0 {
		return false
	}
	if len(toolcall.ParseToolCalls(finalThinking, toolNames)) > 0 {
		return false
	}
	return true
}
