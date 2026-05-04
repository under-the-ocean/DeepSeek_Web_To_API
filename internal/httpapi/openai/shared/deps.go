package shared

import (
	"context"
	"net/http"

	"DeepSeek_Web_To_API/internal/auth"
	"DeepSeek_Web_To_API/internal/chathistory"
	"DeepSeek_Web_To_API/internal/config"
	dsclient "DeepSeek_Web_To_API/internal/deepseek/client"
	"DeepSeek_Web_To_API/internal/util"
)

const (
	// UploadMaxSize limits total multipart request body size (100 MiB).
	UploadMaxSize = 100 << 20
	// GeneralMaxSize limits total JSON request body size (100 MiB).
	GeneralMaxSize = 100 << 20
)

type AuthResolver interface {
	Determine(req *http.Request) (*auth.RequestAuth, error)
	DetermineWithSession(req *http.Request, body []byte) (*auth.RequestAuth, error)
	DetermineCaller(req *http.Request) (*auth.RequestAuth, error)
	Release(a *auth.RequestAuth)
	BindDeepSeekSession(a *auth.RequestAuth, deepseekSessionID string)
	StoreParentMessageID(a *auth.RequestAuth, messageID int)
}

type DeepSeekCaller interface {
	CreateSession(ctx context.Context, a *auth.RequestAuth, maxAttempts int) (string, error)
	GetPow(ctx context.Context, a *auth.RequestAuth, maxAttempts int) (string, error)
	UploadFile(ctx context.Context, a *auth.RequestAuth, req dsclient.UploadFileRequest, maxAttempts int) (*dsclient.UploadFileResult, error)
	CallCompletion(ctx context.Context, a *auth.RequestAuth, payload map[string]any, powResp string, maxAttempts int) (*http.Response, error)
	DeleteSessionForToken(ctx context.Context, token string, sessionID string) (*dsclient.DeleteSessionResult, error)
	DeleteAllSessionsForToken(ctx context.Context, token string) error
}

type ConfigReader interface {
	ModelAliases() map[string]string
	CompatWideInputStrictOutput() bool
	CompatStripReferenceMarkers() bool
	ToolcallMode() string
	ToolcallEarlyEmitConfidence() string
	ResponsesStoreTTLSeconds() int
	EmbeddingsProvider() string
	AutoDeleteMode() string
	AutoDeleteSessions() bool
	HistorySplitEnabled() bool
	HistorySplitTriggerAfterTurns() int
	CurrentInputFileEnabled() bool
	CurrentInputFileMinChars() int
	ThinkingInjectionEnabled() bool
	ThinkingInjectionPrompt() string
}

type Deps struct {
	Store       ConfigReader
	Auth        AuthResolver
	DS          DeepSeekCaller
	ChatHistory *chathistory.Store
}

func CompatStripReferenceMarkers(store ConfigReader) bool {
	if store == nil {
		return true
	}
	return store.CompatStripReferenceMarkers()
}

var WriteJSON = util.WriteJSON

var _ AuthResolver = (*auth.Resolver)(nil)
var _ DeepSeekCaller = (*dsclient.Client)(nil)
var _ ConfigReader = (*config.Store)(nil)
