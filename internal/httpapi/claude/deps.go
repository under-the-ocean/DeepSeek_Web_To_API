package claude

import (
	"context"
	"net/http"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/config"
	dsclient "DeepSeek_Web_To_API/internal/deepseek/client"
)

type AuthResolver interface {
	Determine(req *http.Request) (*auth.RequestAuth, error)
	DetermineWithSession(req *http.Request, body []byte) (*auth.RequestAuth, error)
	Release(a *auth.RequestAuth)
	BindDeepSeekSession(a *auth.RequestAuth, deepseekSessionID string)
	StoreParentMessageID(a *auth.RequestAuth, messageID int)
}

type DeepSeekCaller interface {
	CreateSession(ctx context.Context, a *auth.RequestAuth, maxAttempts int) (string, error)
	GetPow(ctx context.Context, a *auth.RequestAuth, maxAttempts int) (string, error)
	CallCompletion(ctx context.Context, a *auth.RequestAuth, payload map[string]any, powResp string, maxAttempts int) (*http.Response, error)
}

type ConfigReader interface {
	ModelAliases() map[string]string
	CompatStripReferenceMarkers() bool
}

type OpenAIChatRunner interface {
	ChatCompletions(w http.ResponseWriter, r *http.Request)
}

var _ AuthResolver = (*auth.Resolver)(nil)
var _ DeepSeekCaller = (*dsclient.Client)(nil)
var _ ConfigReader = (*config.Store)(nil)
