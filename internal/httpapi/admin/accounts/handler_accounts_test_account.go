package accounts

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	authn "DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
	"DeepSeek_Web_To_API/internal/prompt"
	"DeepSeek_Web_To_API/internal/promptcompat"
	"DeepSeek_Web_To_API/internal/sse"
)

func (h *Handler) testAccount(ctx context.Context, acc config.Account, model, message string) map[string]any {
	start := time.Now()
	identifier := acc.Identifier()
	result := newAccountTestResult(identifier, model, !h.Store.IsEnvBacked())
	defer h.persistAccountTestStatus(identifier, result)

	token, err := h.DS.Login(ctx, acc)
	if err != nil {
		result["message"] = "登录失败: " + err.Error()
		return result
	}
	if err := h.Store.UpdateAccountToken(acc.Identifier(), token); err != nil {
		result["config_warning"] = "登录成功，但 token 持久化失败（仅保存在内存，重启后会丢失）: " + err.Error()
	}

	authCtx := &authn.RequestAuth{UseConfigToken: false, DeepSeekToken: token, AccountID: identifier, Account: acc}
	proxyCtx := authn.WithAuth(ctx, authCtx)
	sessionID, err := h.ensureTestSession(proxyCtx, authCtx, acc, result)
	if err != nil {
		return result
	}
	h.updateSessionCount(proxyCtx, identifier, token, result)

	if strings.TrimSpace(message) == "" {
		result["success"] = true
		result["message"] = withConfigWarning("Token 刷新成功（登录与会话创建成功）", result)
		result["response_time"] = int(time.Since(start).Milliseconds())
		return result
	}
	h.runAccountCompletion(proxyCtx, authCtx, sessionID, model, message, start, result)
	return result
}

func newAccountTestResult(identifier, model string, configWritable bool) map[string]any {
	return map[string]any{
		"account":         identifier,
		"success":         false,
		"response_time":   0,
		"message":         "",
		"model":           model,
		"config_writable": configWritable,
		"config_warning":  "",
	}
}

func (h *Handler) persistAccountTestStatus(identifier string, result map[string]any) {
	status := "failed"
	if ok, _ := result["success"].(bool); ok {
		status = "ok"
	}
	_ = h.Store.UpdateAccountTestStatus(identifier, status)
}

func (h *Handler) ensureTestSession(ctx context.Context, a *authn.RequestAuth, acc config.Account, result map[string]any) (string, error) {
	sessionID, err := h.DS.CreateSession(ctx, a, 1)
	if err == nil {
		return sessionID, nil
	}
	newToken, loginErr := h.DS.Login(ctx, acc)
	if loginErr != nil {
		result["message"] = "创建会话失败: " + err.Error()
		return "", err
	}
	a.DeepSeekToken = newToken
	if err := h.Store.UpdateAccountToken(acc.Identifier(), newToken); err != nil {
		result["config_warning"] = "刷新 token 成功，但 token 持久化失败（仅保存在内存，重启后会丢失）: " + err.Error()
	}
	sessionID, err = h.DS.CreateSession(ctx, a, 1)
	if err != nil {
		result["message"] = "创建会话失败: " + err.Error()
		return "", err
	}
	return sessionID, nil
}

func (h *Handler) updateSessionCount(ctx context.Context, identifier, token string, result map[string]any) {
	sessionStats, sessionErr := h.DS.GetSessionCountForToken(ctx, token)
	if sessionErr != nil || sessionStats == nil {
		return
	}
	sessionCount := sessionStats.FirstPageCount
	result["session_count"] = sessionCount
	_ = h.Store.UpdateAccountSessionCount(identifier, sessionCount)
}

func (h *Handler) runAccountCompletion(ctx context.Context, a *authn.RequestAuth, sessionID string, model, message string, start time.Time, result map[string]any) {
	model, thinking, search := h.resolveAccountTestModel(model)
	pow, err := h.DS.GetPow(ctx, a, 1)
	if err != nil {
		result["message"] = "获取 PoW 失败: " + err.Error()
		return
	}
	payload := promptcompat.StandardRequest{
		ResolvedModel: model,
		FinalPrompt:   prompt.MessagesPrepare([]map[string]any{{"role": "user", "content": message}}),
		Thinking:      thinking,
		Search:        search,
	}.CompletionPayload(sessionID, 0)
	resp, err := h.DS.CallCompletion(ctx, a, payload, pow, 1)
	if err != nil {
		result["message"] = "请求失败: " + err.Error()
		return
	}
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		result["message"] = fmt.Sprintf("请求失败: HTTP %d", resp.StatusCode)
		return
	}
	collected := sse.CollectStream(resp, thinking, true)
	result["success"] = true
	result["response_time"] = int(time.Since(start).Milliseconds())
	if collected.Text != "" {
		result["message"] = collected.Text
	} else {
		result["message"] = "（无回复内容）"
	}
	if collected.Thinking != "" {
		result["thinking"] = collected.Thinking
	}
}

func (h *Handler) resolveAccountTestModel(model string) (string, bool, bool) {
	thinking, search, ok := config.GetModelConfig(model)
	if resolvedModel, resolved := config.ResolveModel(modelAliasSnapshotReader{
		aliases: h.Store.Snapshot().ModelAliases,
	}, model); resolved {
		model = resolvedModel
		thinking, search, ok = config.GetModelConfig(model)
	}
	if !ok {
		return model, false, false
	}
	return model, thinking, search
}

func withConfigWarning(message string, result map[string]any) string {
	warning, _ := result["config_warning"].(string)
	if strings.TrimSpace(warning) == "" {
		return message
	}
	return message + "；" + warning
}
