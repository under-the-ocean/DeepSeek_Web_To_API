package promptcompat

import "DeepSeek_Web_To_API/internal/config"

type StandardRequest struct {
	Surface                 string
	RequestedModel          string
	ResolvedModel           string
	ResponseModel           string
	Messages                []any
	HistoryText             string
	PromptTokenText         string
	CurrentInputFileApplied bool
	ToolsRaw                any
	FinalPrompt             string
	ToolNames               []string
	ToolChoice              ToolChoicePolicy
	Stream                  bool
	Thinking                bool
	ExposeReasoning         bool
	Search                  bool
	RefFileIDs              []string
	RefFileTokens           int
	PassThrough             map[string]any
}

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceForced   ToolChoiceMode = "forced"
)

type ToolChoicePolicy struct {
	Mode       ToolChoiceMode
	ForcedName string
	Allowed    map[string]struct{}
}

func DefaultToolChoicePolicy() ToolChoicePolicy {
	return ToolChoicePolicy{Mode: ToolChoiceAuto}
}

func (p ToolChoicePolicy) IsNone() bool {
	return p.Mode == ToolChoiceNone
}

func (p ToolChoicePolicy) IsRequired() bool {
	return p.Mode == ToolChoiceRequired || p.Mode == ToolChoiceForced
}

func (p ToolChoicePolicy) Allows(name string) bool {
	if len(p.Allowed) == 0 {
		return true
	}
	_, ok := p.Allowed[name]
	return ok
}

func (r StandardRequest) CompletionPayload(sessionID string, parentMessageID int) map[string]any {
	modelID := r.ResolvedModel
	if modelID == "" {
		modelID = r.RequestedModel
	}
	modelType := "default"
	if resolvedType, ok := config.GetModelType(modelID); ok {
		modelType = resolvedType
	}
	refFileIDs := make([]any, 0, len(r.RefFileIDs))
	for _, fileID := range r.RefFileIDs {
		if fileID == "" {
			continue
		}
		refFileIDs = append(refFileIDs, fileID)
	}
	var parentMsgID any = nil
	if parentMessageID > 0 {
		parentMsgID = parentMessageID
	}
	payload := map[string]any{
		"chat_session_id":   sessionID,
		"model_type":        modelType,
		"parent_message_id": parentMsgID,
		"prompt":            r.FinalPrompt,
		"ref_file_ids":      refFileIDs,
		"thinking_enabled":  r.Thinking,
		"search_enabled":    r.Search,
	}
	for k, v := range r.PassThrough {
		payload[k] = v
	}
	return payload
}
