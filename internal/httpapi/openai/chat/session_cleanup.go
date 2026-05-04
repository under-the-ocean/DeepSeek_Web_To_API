package chat

import (
	"context"
	"time"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
)

func (h *Handler) autoDeleteRemoteSession(ctx context.Context, a *auth.RequestAuth, sessionID string) {
	mode := h.Store.AutoDeleteMode()
	if mode == "none" || a.DeepSeekToken == "" {
		return
	}
	if a.DeepSeekSessionID != "" {
		config.Logger.Debug("[auto_delete_sessions] skipped because session is being reused for conversation continuity", "account", a.AccountID, "session_id", sessionID)
		return
	}

	deleteBaseCtx := context.WithoutCancel(ctx)
	deleteCtx, cancel := context.WithTimeout(deleteBaseCtx, 10*time.Second)
	defer cancel()

	switch mode {
	case "single":
		if sessionID == "" {
			config.Logger.Warn("[auto_delete_sessions] skipped single-session delete because session_id is empty", "account", a.AccountID)
			return
		}
		_, err := h.DS.DeleteSessionForToken(deleteCtx, a.DeepSeekToken, sessionID)
		if err != nil {
			config.Logger.Warn("[auto_delete_sessions] failed", "account", a.AccountID, "mode", mode, "session_id", sessionID, "error", err)
			return
		}
		config.Logger.Debug("[auto_delete_sessions] success", "account", a.AccountID, "mode", mode, "session_id", sessionID)
	case "all":
		if err := h.DS.DeleteAllSessionsForToken(deleteCtx, a.DeepSeekToken); err != nil {
			config.Logger.Warn("[auto_delete_sessions] failed", "account", a.AccountID, "mode", mode, "error", err)
			return
		}
		config.Logger.Debug("[auto_delete_sessions] success", "account", a.AccountID, "mode", mode)
	default:
		config.Logger.Warn("[auto_delete_sessions] unknown mode", "account", a.AccountID, "mode", mode)
	}
}
