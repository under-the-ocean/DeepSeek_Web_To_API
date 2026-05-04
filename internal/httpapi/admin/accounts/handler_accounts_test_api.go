package accounts

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	authn "DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/prompt"
	"DeepSeek_Web_To_API/internal/promptcompat"
	"DeepSeek_Web_To_API/internal/sse"
)

func (h *Handler) testAPI(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	_ = json.NewDecoder(r.Body).Decode(&req)
	model, message, apiKey := h.parseAPITestRequest(req)
	if apiKey == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"detail": "没有可用的 API Key"})
		return
	}
	authCtx := &authn.RequestAuth{UseConfigToken: false, DeepSeekToken: apiKey, AccountID: "api-test"}
	proxyCtx := authn.WithAuth(r.Context(), authCtx)

	sessionID, err := h.DS.CreateSession(proxyCtx, authCtx, 1)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"success": false, "error": "创建会话失败: " + err.Error()})
		return
	}
	pow, err := h.DS.GetPow(proxyCtx, authCtx, 1)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"success": false, "error": "获取 PoW 失败: " + err.Error()})
		return
	}
	resolvedModel, thinking, search := h.resolveAccountTestModel(model)
	payload := promptcompat.StandardRequest{
		ResolvedModel: resolvedModel,
		FinalPrompt:   prompt.MessagesPrepare([]map[string]any{{"role": "user", "content": message}}),
		Thinking:      thinking,
		Search:        search,
	}.CompletionPayload(sessionID, 0)
	h.writeAPITestCompletion(w, proxyCtx, authCtx, payload, pow, thinking)
}

func (h *Handler) parseAPITestRequest(req map[string]any) (string, string, string) {
	model, _ := req["model"].(string)
	message := extractRequestMessage(req)
	apiKey, _ := req["api_key"].(string)
	if model == "" {
		model = "deepseek-v4-flash"
	}
	if message == "" {
		message = "你好"
	}
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		keys := h.Store.Snapshot().Keys
		if len(keys) > 0 {
			apiKey = keys[0]
		}
	}
	return model, message, apiKey
}

func (h *Handler) writeAPITestCompletion(w http.ResponseWriter, ctx context.Context, a *authn.RequestAuth, payload map[string]any, pow string, thinking bool) {
	resp, err := h.DS.CallCompletion(ctx, a, payload, pow, 1)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"success": false, "error": "请求失败: " + err.Error()})
		return
	}
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		body, _ := io.ReadAll(resp.Body)
		writeJSON(w, http.StatusOK, map[string]any{"success": false, "status_code": resp.StatusCode, "response": string(body)})
		return
	}
	collected := sse.CollectStream(resp, thinking, true)
	responseBody := map[string]any{"text": "（无回复内容）"}
	if strings.TrimSpace(collected.Text) != "" {
		responseBody["text"] = collected.Text
	}
	if strings.TrimSpace(collected.Thinking) != "" {
		responseBody["thinking"] = collected.Thinking
	}
	writeJSON(w, http.StatusOK, map[string]any{"success": true, "status_code": resp.StatusCode, "response": responseBody})
}

func extractRequestMessage(req map[string]any) string {
	if req == nil {
		return ""
	}
	if message, _ := req["message"].(string); strings.TrimSpace(message) != "" {
		return strings.TrimSpace(message)
	}
	messages, ok := req["messages"].([]any)
	if !ok {
		return ""
	}
	for _, item := range messages {
		msg, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if strings.TrimSpace(role) != "user" {
			continue
		}
		content, _ := msg["content"].(string)
		if strings.TrimSpace(content) != "" {
			return strings.TrimSpace(content)
		}
	}
	return ""
}
