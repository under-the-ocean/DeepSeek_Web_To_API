package account

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"hash"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Affinity binds a "session key" (a stable fingerprint of a conversation) to a
// specific account. It is in-memory, TTL-evicted lazily on lookup. Used so that
// follow-up requests in the same Claude Code / Codex / OpenAI SDK conversation
// keep landing on the same DeepSeek account, preserving upstream session
// continuity (PoW / chat_session_id / KV-cache hits).
type Affinity struct {
	mu       sync.Mutex
	bindings map[string]*affinityEntry
	locks    map[string]*affinityLock
	ttl      time.Duration
}

type affinityEntry struct {
	accountID       string
	sessionID       string
	parentMessageID int
	expiresAt       time.Time
}

type affinityLock struct {
	mu   sync.Mutex
	refs int
}

const defaultAffinityTTLSeconds = 7200 // 2 hours

func NewAffinity() *Affinity {
	ttl := time.Duration(defaultAffinityTTLSeconds) * time.Second
	if v := strings.TrimSpace(os.Getenv("DEEPSEEK_WEB_TO_API_SESSION_AFFINITY_TTL_SECONDS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			ttl = time.Duration(n) * time.Second
		}
	}
	return &Affinity{
		bindings: make(map[string]*affinityEntry),
		locks:    make(map[string]*affinityLock),
		ttl:      ttl,
	}
}

// Lookup returns the bound account for a session key. Refreshes TTL on hit.
func (a *Affinity) Lookup(key string) string {
	if a == nil || key == "" {
		return ""
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	e, ok := a.bindings[key]
	if !ok {
		return ""
	}
	if time.Now().After(e.expiresAt) {
		delete(a.bindings, key)
		return ""
	}
	e.expiresAt = time.Now().Add(a.ttl)
	return e.accountID
}

// LookupSession returns the bound account, DeepSeek session_id, and parentMessageID for a session key.
// Refreshes TTL on hit. Returns empty/zero values if not found or expired.
func (a *Affinity) LookupSession(key string) (accountID string, sessionID string, parentMessageID int) {
	if a == nil || key == "" {
		return "", "", 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	e, ok := a.bindings[key]
	if !ok {
		return "", "", 0
	}
	if time.Now().After(e.expiresAt) {
		delete(a.bindings, key)
		return "", "", 0
	}
	e.expiresAt = time.Now().Add(a.ttl)
	return e.accountID, e.sessionID, e.parentMessageID
}

// Bind associates a session key with an account, refreshing TTL.
// Preserves any existing DeepSeek session_id and parentMessageID to maintain conversation continuity.
func (a *Affinity) Bind(key, accountID string) {
	if a == nil || key == "" || accountID == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	existingSessionID := ""
	existingParentMessageID := 0
	if e, ok := a.bindings[key]; ok {
		existingSessionID = e.sessionID
		existingParentMessageID = e.parentMessageID
	}
	a.bindings[key] = &affinityEntry{
		accountID:       accountID,
		sessionID:       existingSessionID,
		parentMessageID: existingParentMessageID,
		expiresAt:       time.Now().Add(a.ttl),
	}
}

// BindSession associates a session key with both an account and a DeepSeek session_id.
// This enables conversation continuity: follow-up requests reuse the same upstream session.
// Preserves any existing parentMessageID.
func (a *Affinity) BindSession(key, accountID, deepseekSessionID string) {
	if a == nil || key == "" || accountID == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	existingParentMessageID := 0
	if e, ok := a.bindings[key]; ok {
		existingParentMessageID = e.parentMessageID
	}
	a.bindings[key] = &affinityEntry{
		accountID:       accountID,
		sessionID:       deepseekSessionID,
		parentMessageID: existingParentMessageID,
		expiresAt:       time.Now().Add(a.ttl),
	}
}

// StoreParentMessageID stores the response_message_id from a DeepSeek completion
// so the next request in this conversation can set parent_message_id for chain
// continuity within the same session.
func (a *Affinity) StoreParentMessageID(key string, messageID int) {
	if a == nil || key == "" || messageID <= 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if e, ok := a.bindings[key]; ok {
		e.parentMessageID = messageID
	}
}

// Lock serializes the first acquire/bind path for one session key. Without this
// guard, concurrent first turns can all miss Lookup before any goroutine calls
// Bind, spreading one conversation across multiple upstream accounts.
func (a *Affinity) Lock(key string) func() {
	if a == nil || key == "" {
		return func() {}
	}
	a.mu.Lock()
	l := a.locks[key]
	if l == nil {
		l = &affinityLock{}
		a.locks[key] = l
	}
	l.refs++
	a.mu.Unlock()

	l.mu.Lock()
	return func() {
		l.mu.Unlock()
		a.mu.Lock()
		l.refs--
		if l.refs <= 0 && a.locks[key] == l {
			delete(a.locks, key)
		}
		a.mu.Unlock()
	}
}

// Forget removes a binding (used when the bound account is permanently
// unhealthy or the user explicitly resets a conversation).
func (a *Affinity) Forget(key string) {
	if a == nil || key == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.bindings, key)
}

// Stats returns the current binding count and configured TTL.
func (a *Affinity) Stats() (count int, ttl time.Duration) {
	if a == nil {
		return 0, 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.bindings), a.ttl
}

// SessionKey computes a 16-char hex fingerprint of (callerHash, explicit
// user/session identity, system prompt, first user message). Same conversation
// prefix → same key. Returns "" when the body is unparseable or contains no
// extractable user content.
func SessionKey(callerHash [32]byte, body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var doc map[string]any
	if err := json.Unmarshal(body, &doc); err != nil {
		return ""
	}
	h := sha256.New()
	h.Write(callerHash[:])
	h.Write([]byte{0})
	hashSessionIdentity(doc, h)
	hashSystem(doc, h)
	if !hashFirstUserText(doc, h) {
		// no user content → likely malformed or non-chat request; do not bind
		return ""
	}
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:8])
}

// ScopedSessionKey computes a key from an explicit, caller-local scope. The
// caller hash is always part of the digest so two API keys cannot collide even
// if an adapter uses the same scope label.
func ScopedSessionKey(callerHash [32]byte, scope string) string {
	scope = strings.TrimSpace(scope)
	if scope == "" {
		return ""
	}
	h := sha256.New()
	h.Write(callerHash[:])
	h.Write([]byte{0})
	h.Write([]byte("scope:"))
	h.Write([]byte(scope))
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:8])
}

// CallerHash hashes the bearer / api-key token bytes; it's the same shape used
// by auth for caller_id but exposed here so handlers can compute a session key
// without going through the full auth path.
func CallerHash(token string) [32]byte {
	return sha256.Sum256([]byte(strings.TrimSpace(token)))
}

func hashSessionIdentity(doc map[string]any, h hash.Hash) {
	for _, field := range identityFields() {
		if v := strings.TrimSpace(stringValue(doc[field.key])); v != "" {
			writeIdentityField(h, field.label, v)
		}
	}
	if meta, ok := doc["metadata"].(map[string]any); ok {
		for _, field := range identityFields() {
			if v := strings.TrimSpace(stringValue(meta[field.key])); v != "" {
				writeIdentityField(h, "metadata."+field.label, v)
			}
		}
	}
}

func identityFields() []struct {
	key   string
	label string
} {
	return []struct {
		key   string
		label string
	}{
		{key: "user", label: "user"},
		{key: "user_id", label: "user_id"},
		{key: "userId", label: "user_id"},
		{key: "session_id", label: "session_id"},
		{key: "sessionId", label: "session_id"},
		{key: "conversation_id", label: "conversation_id"},
		{key: "conversationId", label: "conversation_id"},
		{key: "thread_id", label: "thread_id"},
		{key: "threadId", label: "thread_id"},
		{key: "response_id", label: "response_id"},
		{key: "responseId", label: "response_id"},
		{key: "previous_response_id", label: "previous_response_id"},
		{key: "previousResponseId", label: "previous_response_id"},
	}
}

func writeIdentityField(h hash.Hash, label, value string) {
	h.Write([]byte("id:"))
	h.Write([]byte(label))
	h.Write([]byte(":"))
	h.Write([]byte(value))
	h.Write([]byte{0})
}

func hashSystem(doc map[string]any, h hash.Hash) {
	if instr, ok := doc["instructions"].(string); ok && strings.TrimSpace(instr) != "" {
		h.Write([]byte("sys:"))
		h.Write([]byte(strings.TrimSpace(instr)))
		h.Write([]byte{0})
	}
	switch s := doc["system"].(type) {
	case string:
		if s != "" {
			h.Write([]byte("sys:"))
			h.Write([]byte(s))
			h.Write([]byte{0})
		}
	case []any:
		for _, item := range s {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if t, ok := m["text"].(string); ok && t != "" {
				h.Write([]byte("sys:"))
				h.Write([]byte(t))
				h.Write([]byte{0})
			}
		}
	}
	if si, ok := doc["systemInstruction"].(map[string]any); ok {
		if parts, ok := si["parts"].([]any); ok {
			for _, p := range parts {
				if m, ok := p.(map[string]any); ok {
					if t, ok := m["text"].(string); ok && t != "" {
						h.Write([]byte("sys:"))
						h.Write([]byte(t))
						h.Write([]byte{0})
					}
				}
			}
		}
	}
	// OpenAI form: messages[role=system|developer].content (also covers
	// translated-from-Claude bodies that the chat handler sees).
	if msgs, ok := doc["messages"].([]any); ok {
		for _, m := range msgs {
			mm, ok := m.(map[string]any)
			if !ok {
				continue
			}
			role, _ := mm["role"].(string)
			if role != "system" && role != "developer" {
				continue
			}
			if c, ok := mm["content"].(string); ok && c != "" {
				h.Write([]byte("sys:"))
				h.Write([]byte(c))
				h.Write([]byte{0})
				continue
			}
			if ca, ok := mm["content"].([]any); ok {
				for _, item := range ca {
					if im, ok := item.(map[string]any); ok {
						if t, ok := im["text"].(string); ok && t != "" {
							h.Write([]byte("sys:"))
							h.Write([]byte(t))
							h.Write([]byte{0})
						}
					}
				}
			}
		}
	}
	hashInputSystem(doc, h)
}

func hashFirstUserText(doc map[string]any, h hash.Hash) bool {
	if msgs, ok := doc["messages"].([]any); ok {
		for _, m := range msgs {
			mm, ok := m.(map[string]any)
			if !ok {
				continue
			}
			role, _ := mm["role"].(string)
			if role != "user" {
				continue
			}
			switch c := mm["content"].(type) {
			case string:
				if c != "" {
					h.Write([]byte("usr:"))
					h.Write([]byte(c))
					return true
				}
			case []any:
				written := false
				for _, item := range c {
					im, ok := item.(map[string]any)
					if !ok {
						continue
					}
					if t, ok := im["text"].(string); ok && t != "" {
						if !written {
							h.Write([]byte("usr:"))
							written = true
						}
						h.Write([]byte(t))
					}
				}
				if written {
					return true
				}
			}
			break
		}
	}
	if hashResponsesInputFirstUser(doc["input"], h) {
		return true
	}
	if conts, ok := doc["contents"].([]any); ok {
		for _, c := range conts {
			cm, ok := c.(map[string]any)
			if !ok {
				continue
			}
			role, _ := cm["role"].(string)
			if role != "" && role != "user" {
				continue
			}
			parts, ok := cm["parts"].([]any)
			if !ok {
				continue
			}
			written := false
			for _, p := range parts {
				pm, ok := p.(map[string]any)
				if !ok {
					continue
				}
				if t, ok := pm["text"].(string); ok && t != "" {
					if !written {
						h.Write([]byte("usr:"))
						written = true
					}
					h.Write([]byte(t))
				}
			}
			if written {
				return true
			}
			break
		}
	}
	return false
}

func hashInputSystem(doc map[string]any, h hash.Hash) {
	items, ok := doc["input"].([]any)
	if !ok {
		return
	}
	for _, item := range items {
		msg, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role := strings.ToLower(strings.TrimSpace(stringValue(msg["role"])))
		if role != "system" && role != "developer" {
			continue
		}
		if text := responseInputText(msg); text != "" {
			h.Write([]byte("sys:"))
			h.Write([]byte(text))
			h.Write([]byte{0})
		}
	}
}

func hashResponsesInputFirstUser(input any, h hash.Hash) bool {
	switch v := input.(type) {
	case string:
		text := strings.TrimSpace(v)
		if text == "" {
			return false
		}
		h.Write([]byte("usr:"))
		h.Write([]byte(text))
		return true
	case []any:
		for _, item := range v {
			switch x := item.(type) {
			case string:
				text := strings.TrimSpace(x)
				if text == "" {
					continue
				}
				h.Write([]byte("usr:"))
				h.Write([]byte(text))
				return true
			case map[string]any:
				role := strings.ToLower(strings.TrimSpace(stringValue(x["role"])))
				if role != "" && role != "user" {
					continue
				}
				if text := responseInputText(x); text != "" {
					h.Write([]byte("usr:"))
					h.Write([]byte(text))
					return true
				}
			}
		}
	case map[string]any:
		if text := responseInputText(v); text != "" {
			h.Write([]byte("usr:"))
			h.Write([]byte(text))
			return true
		}
	}
	return false
}

func responseInputText(msg map[string]any) string {
	if text := strings.TrimSpace(stringValue(msg["text"])); text != "" {
		return text
	}
	switch c := msg["content"].(type) {
	case string:
		return strings.TrimSpace(c)
	case []any:
		var b strings.Builder
		for _, item := range c {
			block, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if text := strings.TrimSpace(stringValue(block["text"])); text != "" {
				b.WriteString(text)
				b.WriteByte('\n')
			}
		}
		return strings.TrimSpace(b.String())
	}
	return ""
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}
