package auth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"DeepSeek_Web_To_API/internal/account"
	"DeepSeek_Web_To_API/internal/config"
)

type ctxKey string

const authCtxKey ctxKey = "auth_context"

const (
	TargetAccountHeader         = "X-DeepSeek-Web-To-API-Target-Account"
	LegacyTargetAccountHeader   = "X-Ds2-Target-Account"
	SessionAffinityHeader       = "X-DeepSeek-Web-To-API-Session-Affinity-Key"
	LegacySessionAffinityHeader = "X-Ds2-Session-Affinity-Key"
)

var (
	ErrUnauthorized       = errors.New("unauthorized: missing auth token")
	ErrNoAccount          = errors.New("no accounts configured or all accounts are busy")
	ErrInvalidDirectToken = errors.New("invalid direct token")
)

const (
	invalidDirectTokenTTL     = 10 * time.Minute
	maxInvalidDirectTokenKeys = 4096
)

type RequestAuth struct {
	UseConfigToken    bool
	DeepSeekToken     string
	DeepSeekSessionID string
	ParentMessageID   int
	CallerID          string
	AccountID         string
	Account           config.Account
	TriedAccounts     map[string]bool
	SessionKey        string
	resolver          *Resolver
}

type LoginFunc func(ctx context.Context, acc config.Account) (string, error)

type Resolver struct {
	Store *config.Store
	Pool  *account.Pool
	Login LoginFunc

	mu               sync.Mutex
	tokenRefreshedAt map[string]time.Time
	invalidDirectAt  map[string]time.Time
}

func NewResolver(store *config.Store, pool *account.Pool, login LoginFunc) *Resolver {
	return &Resolver{
		Store:            store,
		Pool:             pool,
		Login:            login,
		tokenRefreshedAt: map[string]time.Time{},
		invalidDirectAt:  map[string]time.Time{},
	}
}

func (r *Resolver) Determine(req *http.Request) (*RequestAuth, error) {
	callerKey := extractCallerToken(req)
	if callerKey == "" {
		return nil, ErrUnauthorized
	}
	callerID := callerTokenID(callerKey)
	ctx := req.Context()
	if !r.Store.HasAPIKey(callerKey) {
		if r.directTokenBlocked(callerID, time.Now()) {
			return nil, ErrInvalidDirectToken
		}
		return &RequestAuth{
			UseConfigToken: false,
			DeepSeekToken:  callerKey,
			CallerID:       callerID,
			resolver:       r,
			TriedAccounts:  map[string]bool{},
		}, nil
	}
	target := requestHeader(req, TargetAccountHeader, LegacyTargetAccountHeader)
	a, err := r.acquireManagedRequestAuth(ctx, callerID, target, false)
	if err != nil {
		return nil, err
	}
	return a, nil
}

// DetermineWithSession is Determine + session affinity. If body parses as a
// chat-style request, we compute a stable per-conversation key from
// (caller_token, system, first user message); on a hit the bound account is
// injected into X-DeepSeek-Web-To-API-Target-Account so subsequent turns of
// the same Claude Code / Codex / OpenAI conversation land on the same DeepSeek
// account.
// On successful managed acquire, the binding is created/refreshed.
// Additionally, if a DeepSeek session_id was previously bound, it is returned
// in RequestAuth.DeepSeekSessionID for conversation continuity, and the
// parent_message_id is returned in RequestAuth.ParentMessageID so the next
// completion payload can chain within the same upstream session.
func (r *Resolver) DetermineWithSession(req *http.Request, body []byte) (*RequestAuth, error) {
	callerKey := extractCallerToken(req)
	if callerKey == "" {
		return nil, ErrUnauthorized
	}
	callerID := callerTokenID(callerKey)
	if !r.Store.HasAPIKey(callerKey) {
		if r.directTokenBlocked(callerID, time.Now()) {
			return nil, ErrInvalidDirectToken
		}
		return &RequestAuth{
			UseConfigToken: false,
			DeepSeekToken:  callerKey,
			CallerID:       callerID,
			resolver:       r,
			TriedAccounts:  map[string]bool{},
		}, nil
	}

	var sessionKey string
	var existingSessionID string
	var existingParentMessageID int
	target := requestHeader(req, TargetAccountHeader, LegacyTargetAccountHeader)
	explicitTarget := target != ""
	if len(body) > 0 && r.Pool != nil && r.Pool.Affinity != nil {
		callerHash := account.CallerHash(callerKey)
		sessionKey = sessionKeyFromRequest(req, callerHash, body)
		unlock := r.Pool.Affinity.Lock(sessionKey)
		defer unlock()
		if sessionKey != "" && target == "" {
			boundAccount, boundSession, boundParentID := r.Pool.Affinity.LookupSession(sessionKey)
			if boundAccount != "" {
				target = boundAccount
			}
			if boundSession != "" {
				existingSessionID = boundSession
			}
			if boundParentID > 0 {
				existingParentMessageID = boundParentID
			}
		}
	}

	a, err := r.acquireManagedRequestAuth(req.Context(), callerID, target, sessionKey != "" && !explicitTarget)
	if err != nil {
		return nil, err
	}
	if a != nil {
		a.SessionKey = sessionKey
		a.DeepSeekSessionID = existingSessionID
		a.ParentMessageID = existingParentMessageID
	}
	if sessionKey != "" && a != nil && a.UseConfigToken && a.AccountID != "" && r.Pool != nil && r.Pool.Affinity != nil {
		r.Pool.Affinity.Bind(sessionKey, a.AccountID)
	}
	return a, nil
}

func sessionKeyFromRequest(req *http.Request, callerHash [32]byte, body []byte) string {
	if req != nil {
		if scope := requestHeader(req, SessionAffinityHeader, LegacySessionAffinityHeader); scope != "" {
			return account.ScopedSessionKey(callerHash, scope)
		}
	}
	bodyKey := account.SessionKey(callerHash, body)
	if bodyKey == "" {
		return ""
	}
	if req != nil && req.URL != nil {
		if path := strings.TrimSpace(req.URL.Path); path != "" {
			return account.ScopedSessionKey(callerHash, "http:"+path+":body:"+bodyKey)
		}
	}
	return bodyKey
}

func requestHeader(req *http.Request, primary string, legacy ...string) string {
	if req == nil {
		return ""
	}
	if value := strings.TrimSpace(req.Header.Get(primary)); value != "" {
		return value
	}
	for _, name := range legacy {
		if value := strings.TrimSpace(req.Header.Get(name)); value != "" {
			return value
		}
	}
	return ""
}

// SessionAffinity exposes the underlying affinity store for admin endpoints.
func (r *Resolver) SessionAffinity() *account.Affinity {
	if r == nil || r.Pool == nil {
		return nil
	}
	return r.Pool.Affinity
}

func (r *Resolver) acquireManagedRequestAuth(ctx context.Context, callerID, target string, allowOverflow bool) (*RequestAuth, error) {
	tried := map[string]bool{}
	var lastEnsureErr error
	for {
		if (target == "" || allowOverflow) && len(tried) >= len(r.Store.Accounts()) {
			if lastEnsureErr != nil {
				return nil, lastEnsureErr
			}
			return nil, ErrNoAccount
		}
		var acc config.Account
		var ok bool
		if allowOverflow {
			acc, ok = r.Pool.AcquireWaitPreferred(ctx, target, tried)
		} else {
			acc, ok = r.Pool.AcquireWait(ctx, target, tried)
		}
		if !ok {
			if lastEnsureErr != nil {
				return nil, lastEnsureErr
			}
			return nil, ErrNoAccount
		}

		a := &RequestAuth{
			UseConfigToken: true,
			CallerID:       callerID,
			AccountID:      acc.Identifier(),
			Account:        acc,
			TriedAccounts:  tried,
			resolver:       r,
		}

		if err := r.ensureManagedToken(ctx, a); err != nil {
			lastEnsureErr = err
			tried[a.AccountID] = true
			r.Pool.Release(a.AccountID)
			if target != "" && !allowOverflow {
				return nil, err
			}
			continue
		}
		return a, nil
	}
}

// DetermineCaller resolves caller identity without acquiring any pooled account.
// Use this for local-cache lookup routes that only need tenant isolation.
func (r *Resolver) DetermineCaller(req *http.Request) (*RequestAuth, error) {
	callerKey := extractCallerToken(req)
	if callerKey == "" {
		return nil, ErrUnauthorized
	}
	callerID := callerTokenID(callerKey)
	a := &RequestAuth{
		UseConfigToken: false,
		CallerID:       callerID,
		resolver:       r,
		TriedAccounts:  map[string]bool{},
	}
	if r == nil || r.Store == nil || !r.Store.HasAPIKey(callerKey) {
		if r != nil && r.directTokenBlocked(callerID, time.Now()) {
			return nil, ErrInvalidDirectToken
		}
		a.DeepSeekToken = callerKey
	}
	return a, nil
}

func WithAuth(ctx context.Context, a *RequestAuth) context.Context {
	return context.WithValue(ctx, authCtxKey, a)
}

func FromContext(ctx context.Context) (*RequestAuth, bool) {
	v := ctx.Value(authCtxKey)
	a, ok := v.(*RequestAuth)
	return a, ok
}

func (r *Resolver) loginAndPersist(ctx context.Context, a *RequestAuth) error {
	token, err := r.Login(ctx, a.Account)
	if err != nil {
		return err
	}
	a.Account.Token = token
	a.DeepSeekToken = token
	r.markTokenRefreshedNow(a.AccountID)
	return r.Store.UpdateAccountToken(a.AccountID, token)
}

func (r *Resolver) RefreshToken(ctx context.Context, a *RequestAuth) bool {
	if !a.UseConfigToken || a.AccountID == "" {
		return false
	}
	_ = r.Store.UpdateAccountToken(a.AccountID, "")
	a.Account.Token = ""
	if err := r.loginAndPersist(ctx, a); err != nil {
		config.Logger.Error("[refresh_token] failed", "account", a.AccountID, "error", err)
		return false
	}
	return true
}

func (r *Resolver) MarkTokenInvalid(a *RequestAuth) {
	if !a.UseConfigToken || a.AccountID == "" {
		return
	}
	a.Account.Token = ""
	a.DeepSeekToken = ""
	r.clearTokenRefreshMark(a.AccountID)
	_ = r.Store.UpdateAccountToken(a.AccountID, "")
}

func (r *Resolver) MarkDirectTokenInvalid(a *RequestAuth) {
	if r == nil || a == nil || a.UseConfigToken {
		return
	}
	callerID := strings.TrimSpace(a.CallerID)
	if callerID == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.invalidDirectAt == nil {
		r.invalidDirectAt = map[string]time.Time{}
	}
	now := time.Now()
	r.pruneInvalidDirectLocked(now)
	if len(r.invalidDirectAt) >= maxInvalidDirectTokenKeys {
		r.evictOldestInvalidDirectLocked()
	}
	r.invalidDirectAt[callerID] = now
}

func (a *RequestAuth) MarkDirectTokenInvalid() {
	if a == nil || a.resolver == nil {
		return
	}
	a.resolver.MarkDirectTokenInvalid(a)
}

func (r *Resolver) directTokenBlocked(callerID string, now time.Time) bool {
	if r == nil {
		return false
	}
	callerID = strings.TrimSpace(callerID)
	if callerID == "" {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.invalidDirectAt == nil {
		return false
	}
	markedAt, ok := r.invalidDirectAt[callerID]
	if !ok {
		return false
	}
	if now.Sub(markedAt) >= invalidDirectTokenTTL {
		delete(r.invalidDirectAt, callerID)
		return false
	}
	return true
}

func (r *Resolver) pruneInvalidDirectLocked(now time.Time) {
	for callerID, markedAt := range r.invalidDirectAt {
		if now.Sub(markedAt) >= invalidDirectTokenTTL {
			delete(r.invalidDirectAt, callerID)
		}
	}
}

func (r *Resolver) evictOldestInvalidDirectLocked() {
	var oldestCallerID string
	var oldestMarkedAt time.Time
	for callerID, markedAt := range r.invalidDirectAt {
		if oldestCallerID == "" || markedAt.Before(oldestMarkedAt) {
			oldestCallerID = callerID
			oldestMarkedAt = markedAt
		}
	}
	if oldestCallerID != "" {
		delete(r.invalidDirectAt, oldestCallerID)
	}
}

func (r *Resolver) SwitchAccount(ctx context.Context, a *RequestAuth) bool {
	if !a.UseConfigToken {
		return false
	}
	if a.TriedAccounts == nil {
		a.TriedAccounts = map[string]bool{}
	}
	if a.AccountID != "" {
		a.TriedAccounts[a.AccountID] = true
		r.Pool.Release(a.AccountID)
	}
	for {
		acc, ok := r.Pool.AcquireWait(ctx, "", a.TriedAccounts)
		if !ok {
			return false
		}
		a.Account = acc
		a.AccountID = acc.Identifier()
		if err := r.ensureManagedToken(ctx, a); err != nil {
			a.TriedAccounts[a.AccountID] = true
			r.Pool.Release(a.AccountID)
			continue
		}
		r.bindSessionAffinity(a)
		return true
	}
}

func (r *Resolver) bindSessionAffinity(a *RequestAuth) {
	if r == nil || r.Pool == nil || r.Pool.Affinity == nil || a == nil {
		return
	}
	if a.SessionKey == "" || a.AccountID == "" {
		return
	}
	r.Pool.Affinity.Bind(a.SessionKey, a.AccountID)
}

// BindDeepSeekSession associates a DeepSeek session_id with the current session key.
// This enables conversation continuity: follow-up requests can reuse the same upstream session.
func (r *Resolver) BindDeepSeekSession(a *RequestAuth, deepseekSessionID string) {
	if r == nil || r.Pool == nil || r.Pool.Affinity == nil || a == nil {
		return
	}
	if a.SessionKey == "" || a.AccountID == "" || deepseekSessionID == "" {
		return
	}
	r.Pool.Affinity.BindSession(a.SessionKey, a.AccountID, deepseekSessionID)
}

// StoreParentMessageID persists the response_message_id from a DeepSeek completion
// so the next request in this conversation can set parent_message_id for chain continuity.
func (r *Resolver) StoreParentMessageID(a *RequestAuth, messageID int) {
	if r == nil || r.Pool == nil || r.Pool.Affinity == nil || a == nil {
		return
	}
	if a.SessionKey == "" || messageID <= 0 {
		return
	}
	r.Pool.Affinity.StoreParentMessageID(a.SessionKey, messageID)
}

func (r *Resolver) Release(a *RequestAuth) {
	if a == nil || !a.UseConfigToken || a.AccountID == "" {
		return
	}
	r.Pool.Release(a.AccountID)
}

func extractCallerToken(req *http.Request) string {
	authHeader := strings.TrimSpace(req.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
		token := strings.TrimSpace(authHeader[7:])
		if token != "" {
			return token
		}
	}
	if key := strings.TrimSpace(req.Header.Get("x-api-key")); key != "" {
		return key
	}
	// Gemini/Google clients commonly send API key via x-goog-api-key.
	if key := strings.TrimSpace(req.Header.Get("x-goog-api-key")); key != "" {
		return key
	}
	// Query key fallback is disabled by default for security reasons.
	// Enable with DEEPSEEK_WEB_TO_API_ALLOW_QUERY_AUTH=true if you have legacy clients.
	if !allowQueryTokenAuth() {
		return ""
	}
	if key := strings.TrimSpace(req.URL.Query().Get("key")); key != "" {
		return key
	}
	return strings.TrimSpace(req.URL.Query().Get("api_key"))
}

func allowQueryTokenAuth() bool {
	v := strings.TrimSpace(os.Getenv("DEEPSEEK_WEB_TO_API_ALLOW_QUERY_AUTH"))
	if v == "" {
		return false
	}
	enabled, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return enabled
}

func callerTokenID(token string) string {
	token = strings.TrimSpace(token)
	if token == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(token))
	return "caller:" + hex.EncodeToString(sum[:8])
}

func (r *Resolver) ensureManagedToken(ctx context.Context, a *RequestAuth) error {
	if strings.TrimSpace(a.Account.Token) == "" {
		return r.loginAndPersist(ctx, a)
	}
	if r.shouldForceRefresh(a.AccountID) {
		if err := r.loginAndPersist(ctx, a); err != nil {
			return err
		}
		return nil
	}
	a.DeepSeekToken = a.Account.Token
	return nil
}

func (r *Resolver) shouldForceRefresh(accountID string) bool {
	if r == nil || r.Store == nil {
		return false
	}
	if strings.TrimSpace(accountID) == "" {
		return false
	}
	intervalHours := r.Store.RuntimeTokenRefreshIntervalHours()
	if intervalHours <= 0 {
		return false
	}
	now := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()
	last, ok := r.tokenRefreshedAt[accountID]
	if !ok || last.IsZero() {
		r.tokenRefreshedAt[accountID] = now
		return false
	}
	return now.Sub(last) >= time.Duration(intervalHours)*time.Hour
}

func (r *Resolver) markTokenRefreshedNow(accountID string) {
	if strings.TrimSpace(accountID) == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tokenRefreshedAt[accountID] = time.Now()
}

func (r *Resolver) clearTokenRefreshMark(accountID string) {
	if strings.TrimSpace(accountID) == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.tokenRefreshedAt, accountID)
}
