package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"charm.land/fantasy/schema"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const topLogprobsMax = 20

type responsesLanguageModel struct {
	provider           string
	modelID            string
	client             openai.Client
	objectMode         fantasy.ObjectMode
	reasoningModelFunc func(modelID string) bool
}

// newResponsesLanguageModel implements a responses api model.
func newResponsesLanguageModel(modelID string, provider string, client openai.Client, objectMode fantasy.ObjectMode, reasoningModelFunc func(modelID string) bool) responsesLanguageModel {
	return responsesLanguageModel{
		modelID:            modelID,
		provider:           provider,
		client:             client,
		objectMode:         objectMode,
		reasoningModelFunc: reasoningModelFunc,
	}
}

func (o responsesLanguageModel) Model() string {
	return o.modelID
}

func (o responsesLanguageModel) Provider() string {
	return o.provider
}

type responsesModelConfig struct {
	isReasoningModel           bool
	systemMessageMode          string
	requiredAutoTruncation     bool
	supportsFlexProcessing     bool
	supportsPriorityProcessing bool
}

func getResponsesModelConfig(modelID string) responsesModelConfig {
	supportsFlexProcessing := strings.HasPrefix(modelID, "o3") ||
		strings.Contains(modelID, "-o3") || strings.Contains(modelID, "o4-mini") ||
		(strings.Contains(modelID, "gpt-5") && !strings.Contains(modelID, "gpt-5-chat"))

	supportsPriorityProcessing := strings.Contains(strings.ToLower(modelID), "gpt-4") ||
		strings.Contains(strings.ToLower(modelID), "gpt-5-mini") ||
		(strings.Contains(strings.ToLower(modelID), "gpt-5") &&
			!strings.Contains(strings.ToLower(modelID), "gpt-5-nano") &&
			!strings.Contains(strings.ToLower(modelID), "gpt-5-chat")) ||
		strings.HasPrefix(modelID, "o3") ||
		strings.Contains(modelID, "-o3") ||
		strings.Contains(modelID, "o4-mini")

	defaults := responsesModelConfig{
		requiredAutoTruncation:     false,
		systemMessageMode:          "system",
		supportsFlexProcessing:     supportsFlexProcessing,
		supportsPriorityProcessing: supportsPriorityProcessing,
	}

	reasoningGeneration := reasoningGenerationPattern.MatchString(strings.ToLower(modelID))
	if reasoningGeneration && strings.Contains(strings.ToLower(modelID), "-chat") {
		return responsesModelConfig{
			isReasoningModel:           false,
			systemMessageMode:          defaults.systemMessageMode,
			requiredAutoTruncation:     defaults.requiredAutoTruncation,
			supportsFlexProcessing:     defaults.supportsFlexProcessing,
			supportsPriorityProcessing: defaults.supportsPriorityProcessing,
		}
	}

	if strings.HasPrefix(modelID, "o1") || strings.Contains(modelID, "-o1") ||
		strings.HasPrefix(modelID, "o3") || strings.Contains(modelID, "-o3") ||
		strings.HasPrefix(modelID, "o4") || strings.Contains(modelID, "-o4") ||
		strings.HasPrefix(modelID, "oss") || strings.Contains(modelID, "-oss") ||
		reasoningGeneration ||
		strings.Contains(modelID, "codex-") || strings.Contains(modelID, "computer-use") {
		if strings.Contains(modelID, "o1-mini") || strings.Contains(modelID, "o1-preview") {
			return responsesModelConfig{
				isReasoningModel:           true,
				systemMessageMode:          "remove",
				requiredAutoTruncation:     defaults.requiredAutoTruncation,
				supportsFlexProcessing:     defaults.supportsFlexProcessing,
				supportsPriorityProcessing: defaults.supportsPriorityProcessing,
			}
		}

		return responsesModelConfig{
			isReasoningModel:           true,
			systemMessageMode:          "developer",
			requiredAutoTruncation:     defaults.requiredAutoTruncation,
			supportsFlexProcessing:     defaults.supportsFlexProcessing,
			supportsPriorityProcessing: defaults.supportsPriorityProcessing,
		}
	}

	return responsesModelConfig{
		isReasoningModel:           false,
		systemMessageMode:          defaults.systemMessageMode,
		requiredAutoTruncation:     defaults.requiredAutoTruncation,
		supportsFlexProcessing:     defaults.supportsFlexProcessing,
		supportsPriorityProcessing: defaults.supportsPriorityProcessing,
	}
}

const (
	previousResponseIDHistoryError = "cannot combine previous_response_id with replayed conversation history; use either previous_response_id (server-side chaining) or explicit message replay, not both"
	previousResponseIDStoreError   = "previous_response_id requires store to be true; the current response will not be stored and cannot be used for further chaining"
)

func (o responsesLanguageModel) prepareParams(call fantasy.Call) (*responses.ResponseNewParams, []fantasy.CallWarning, error) {
	var warnings []fantasy.CallWarning
	params := &responses.ResponseNewParams{}

	modelConfig := getResponsesModelConfig(o.modelID)
	if o.reasoningModelFunc != nil {
		if reasoning := o.reasoningModelFunc(o.modelID); reasoning != modelConfig.isReasoningModel {
			modelConfig.isReasoningModel = reasoning
			modelConfig.systemMessageMode = "system"
			if reasoning {
				modelConfig.systemMessageMode = "developer"
			}
		}
	}

	if call.TopK != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "topK",
		})
	}

	if call.PresencePenalty != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "presencePenalty",
		})
	}

	if call.FrequencyPenalty != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "frequencyPenalty",
		})
	}

	var openaiOptions *ResponsesProviderOptions
	if opts, ok := call.ProviderOptions[Name]; ok {
		if typedOpts, ok := opts.(*ResponsesProviderOptions); ok {
			openaiOptions = typedOpts
		}
	}

	if openaiOptions != nil && openaiOptions.Store != nil {
		params.Store = param.NewOpt(*openaiOptions.Store)
	} else {
		params.Store = param.NewOpt(false)
	}

	allowPreviousResponseToolOutputs := false
	if openaiOptions != nil && openaiOptions.PreviousResponseID != nil && *openaiOptions.PreviousResponseID != "" {
		if err := validatePreviousResponseIDPrompt(call.Prompt); err != nil {
			return nil, warnings, err
		}
		if openaiOptions.Store == nil || !*openaiOptions.Store {
			return nil, warnings, errors.New(previousResponseIDStoreError)
		}
		params.PreviousResponseID = param.NewOpt(*openaiOptions.PreviousResponseID)
		allowPreviousResponseToolOutputs = true
	}

	storeEnabled := openaiOptions != nil && openaiOptions.Store != nil && *openaiOptions.Store
	var input responses.ResponseInputParam
	var inputWarnings []fantasy.CallWarning
	var err error
	if allowPreviousResponseToolOutputs {
		input, inputWarnings, err = toResponsesPromptWithValidation(call.Prompt, modelConfig.systemMessageMode, storeEnabled, true)
	} else {
		input, inputWarnings, err = toResponsesPrompt(call.Prompt, modelConfig.systemMessageMode, storeEnabled)
	}
	warnings = append(warnings, inputWarnings...)
	if err != nil {
		return nil, warnings, err
	}

	var include []IncludeType

	addInclude := func(key IncludeType) {
		include = append(include, key)
	}

	topLogprobs := 0
	if openaiOptions != nil && openaiOptions.Logprobs != nil {
		switch v := openaiOptions.Logprobs.(type) {
		case bool:
			if v {
				topLogprobs = topLogprobsMax
			}
		case float64:
			topLogprobs = int(v)
		case int:
			topLogprobs = v
		}
	}

	if topLogprobs > 0 {
		addInclude(IncludeMessageOutputTextLogprobs)
	}

	params.Model = o.modelID
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	if call.Temperature != nil {
		params.Temperature = param.NewOpt(*call.Temperature)
	}
	if call.TopP != nil {
		params.TopP = param.NewOpt(*call.TopP)
	}
	if call.MaxOutputTokens != nil {
		params.MaxOutputTokens = param.NewOpt(*call.MaxOutputTokens)
	}

	if openaiOptions != nil {
		if openaiOptions.MaxToolCalls != nil {
			params.MaxToolCalls = param.NewOpt(*openaiOptions.MaxToolCalls)
		}
		if openaiOptions.Metadata != nil {
			metadata := make(shared.Metadata)
			for k, v := range openaiOptions.Metadata {
				if str, ok := v.(string); ok {
					metadata[k] = str
				}
			}
			params.Metadata = metadata
		}
		if openaiOptions.ParallelToolCalls != nil {
			params.ParallelToolCalls = param.NewOpt(*openaiOptions.ParallelToolCalls)
		}
		if openaiOptions.User != nil {
			params.User = param.NewOpt(*openaiOptions.User)
		}
		if openaiOptions.Instructions != nil {
			params.Instructions = param.NewOpt(*openaiOptions.Instructions)
		}
		if openaiOptions.ServiceTier != nil {
			params.ServiceTier = responses.ResponseNewParamsServiceTier(*openaiOptions.ServiceTier)
		}
		if openaiOptions.PromptCacheKey != nil {
			params.PromptCacheKey = param.NewOpt(*openaiOptions.PromptCacheKey)
		}
		if openaiOptions.SafetyIdentifier != nil {
			params.SafetyIdentifier = param.NewOpt(*openaiOptions.SafetyIdentifier)
		}
		if topLogprobs > 0 {
			params.TopLogprobs = param.NewOpt(int64(topLogprobs))
		}

		if len(openaiOptions.Include) > 0 {
			include = append(include, openaiOptions.Include...)
		}

		if modelConfig.isReasoningModel && (openaiOptions.ReasoningEffort != nil || openaiOptions.ReasoningSummary != nil) {
			reasoning := shared.ReasoningParam{}
			if openaiOptions.ReasoningEffort != nil {
				reasoning.Effort = shared.ReasoningEffort(*openaiOptions.ReasoningEffort)
			}
			if openaiOptions.ReasoningSummary != nil {
				reasoning.Summary = shared.ReasoningSummary(*openaiOptions.ReasoningSummary)
			}
			params.Reasoning = reasoning
		}
	}

	if modelConfig.requiredAutoTruncation {
		params.Truncation = responses.ResponseNewParamsTruncationAuto
	}

	if len(include) > 0 {
		includeParams := make([]responses.ResponseIncludable, len(include))
		for i, inc := range include {
			includeParams[i] = responses.ResponseIncludable(string(inc))
		}
		params.Include = includeParams
	}

	if modelConfig.isReasoningModel {
		if call.Temperature != nil {
			params.Temperature = param.Opt[float64]{}
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "temperature",
				Details: "temperature is not supported for reasoning models",
			})
		}

		if call.TopP != nil {
			params.TopP = param.Opt[float64]{}
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "topP",
				Details: "topP is not supported for reasoning models",
			})
		}
	} else {
		if openaiOptions != nil {
			if openaiOptions.ReasoningEffort != nil {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeUnsupportedSetting,
					Setting: "reasoningEffort",
					Details: "reasoningEffort is not supported for non-reasoning models",
				})
			}

			if openaiOptions.ReasoningSummary != nil {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeUnsupportedSetting,
					Setting: "reasoningSummary",
					Details: "reasoningSummary is not supported for non-reasoning models",
				})
			}
		}
	}

	if openaiOptions != nil && openaiOptions.ServiceTier != nil {
		if *openaiOptions.ServiceTier == ServiceTierFlex && !modelConfig.supportsFlexProcessing {
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "serviceTier",
				Details: "flex processing is only available for o3, o4-mini, and gpt-5 models",
			})
			params.ServiceTier = ""
		}

		if *openaiOptions.ServiceTier == ServiceTierPriority && !modelConfig.supportsPriorityProcessing {
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "serviceTier",
				Details: "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported",
			})
			params.ServiceTier = ""
		}
	}

	tools, toolChoice, toolWarnings := toResponsesTools(call.Tools, call.ToolChoice, openaiOptions)
	warnings = append(warnings, toolWarnings...)

	if len(tools) > 0 {
		params.Tools = tools
		params.ToolChoice = toolChoice
	}

	return params, warnings, nil
}

func validatePreviousResponseIDPrompt(prompt fantasy.Prompt) error {
	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem, fantasy.MessageRoleUser, fantasy.MessageRoleTool:
			// System and user messages are baseline inputs.
			// Tool messages carry computer_call_output and
			// function call_output items, which the OpenAI
			// Responses API explicitly supports as the
			// follow-up payload alongside previous_response_id
			continue
		default:
			return errors.New(previousResponseIDHistoryError)
		}
	}
	return nil
}

func responsesProviderMetadata(responseID string) fantasy.ProviderMetadata {
	if responseID == "" {
		return fantasy.ProviderMetadata{}
	}

	return fantasy.ProviderMetadata{
		Name: &ResponsesProviderMetadata{
			ResponseID: responseID,
		},
	}
}

func responsesUsage(resp responses.Response) fantasy.Usage {
	// OpenAI reports input_tokens INCLUDING cached tokens. Subtract to avoid double-counting.
	inputTokens := max(resp.Usage.InputTokens-resp.Usage.InputTokensDetails.CachedTokens, 0)
	usage := fantasy.Usage{
		InputTokens:  inputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
	}
	if resp.Usage.OutputTokensDetails.ReasoningTokens != 0 {
		usage.ReasoningTokens = resp.Usage.OutputTokensDetails.ReasoningTokens
	}
	if resp.Usage.InputTokensDetails.CachedTokens != 0 {
		usage.CacheReadTokens = resp.Usage.InputTokensDetails.CachedTokens
	}
	return usage
}

func toResponsesPrompt(prompt fantasy.Prompt, systemMessageMode string, store bool) (responses.ResponseInputParam, []fantasy.CallWarning, error) {
	return toResponsesPromptWithValidation(prompt, systemMessageMode, store, false)
}

func toResponsesPromptWithValidation(prompt fantasy.Prompt, systemMessageMode string, store bool, allowUnpairedFunctionCallOutputs bool) (responses.ResponseInputParam, []fantasy.CallWarning, error) {
	var input responses.ResponseInputParam
	var warnings []fantasy.CallWarning

	// First pass: collect raw JSON for computer_call output items.
	// This enables faithful round-tripping via param.Override.
	computerCallRawJSON := make(map[string]string)
	for _, msg := range prompt {
		if msg.Role != fantasy.MessageRoleAssistant {
			continue
		}
		for _, c := range msg.Content {
			if c.GetType() != fantasy.ContentTypeToolCall {
				continue
			}
			toolCallPart, ok := fantasy.AsContentType[fantasy.ToolCallPart](c)
			if !ok {
				continue
			}
			if meta := GetComputerUseMetadata(toolCallPart.ProviderOptions); meta != nil && meta.RawJSON != "" {
				computerCallRawJSON[toolCallPart.ToolCallID] = meta.RawJSON
			}
		}
	}

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			var systemText string
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeText {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "system prompt can only have text content",
					})
					continue
				}
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "system prompt text part does not have the right type",
					})
					continue
				}
				if strings.TrimSpace(textPart.Text) != "" {
					systemText += textPart.Text
				}
			}

			if systemText == "" {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "system prompt has no text parts",
				})
				continue
			}

			switch systemMessageMode {
			case "system":
				input = append(input, responses.ResponseInputItemParamOfMessage(systemText, responses.EasyInputMessageRoleSystem))
			case "developer":
				input = append(input, responses.ResponseInputItemParamOfMessage(systemText, responses.EasyInputMessageRoleDeveloper))
			case "remove":
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "system messages are removed for this model",
				})
			}

		case fantasy.MessageRoleUser:
			var contentParts responses.ResponseInputMessageContentListParam
			for i, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "user message text part does not have the right type",
						})
						continue
					}
					contentParts = append(contentParts, responses.ResponseInputContentUnionParam{
						OfInputText: &responses.ResponseInputTextParam{
							Type: "input_text",
							Text: textPart.Text,
						},
					})

				case fantasy.ContentTypeFile:
					filePart, ok := fantasy.AsContentType[fantasy.FilePart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "user message file part does not have the right type",
						})
						continue
					}

					if strings.HasPrefix(filePart.MediaType, "image/") {
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						imageURL := fmt.Sprintf("data:%s;base64,%s", filePart.MediaType, base64Encoded)
						contentParts = append(contentParts, responses.ResponseInputContentUnionParam{
							OfInputImage: &responses.ResponseInputImageParam{
								Type:     "input_image",
								ImageURL: param.NewOpt(imageURL),
							},
						})
					} else if filePart.MediaType == "application/pdf" {
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						fileData := fmt.Sprintf("data:application/pdf;base64,%s", base64Encoded)
						filename := filePart.Filename
						if filename == "" {
							filename = fmt.Sprintf("part-%d.pdf", i)
						}
						contentParts = append(contentParts, responses.ResponseInputContentUnionParam{
							OfInputFile: &responses.ResponseInputFileParam{
								Type:     "input_file",
								Filename: param.NewOpt(filename),
								FileData: param.NewOpt(fileData),
							},
						})
					} else {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}

			if !hasVisibleResponsesUserContent(contentParts) {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "dropping empty user message (contains neither user-facing content nor tool results)",
				})
				continue
			}

			input = append(input, responses.ResponseInputItemParamOfMessage(contentParts, responses.EasyInputMessageRoleUser))

		case fantasy.MessageRoleAssistant:
			startIdx := len(input)
			lastEmittedReasoningReference := false
			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message text part does not have the right type",
						})
						continue
					}
					input = append(input, responses.ResponseInputItemParamOfMessage(textPart.Text, responses.EasyInputMessageRoleAssistant))
					lastEmittedReasoningReference = false

				case fantasy.ContentTypeToolCall:
					toolCallPart, ok := fantasy.AsContentType[fantasy.ToolCallPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message tool call part does not have the right type",
						})
						continue
					}

					if toolCallPart.ProviderExecuted {
						if store && lastEmittedReasoningReference &&
							isResponsesWebSearchToolCall(toolCallPart) &&
							toolCallPart.ToolCallID != "" {
							input = append(input, responses.ResponseInputItemParamOfItemReference(toolCallPart.ToolCallID))
						}
						lastEmittedReasoningReference = false
						continue
					}

					// Computer calls are round-tripped via param.Override
					// using the raw wire-format JSON from the original response.
					if rawJSON, ok := computerCallRawJSON[toolCallPart.ToolCallID]; ok {
						overridden := param.Override[responses.ResponseComputerToolCallParam](json.RawMessage(rawJSON))
						input = append(input, responses.ResponseInputItemUnionParam{
							OfComputerCall: &overridden,
						})
						continue
					}

					input = append(input, responses.ResponseInputItemParamOfFunctionCall(toolCallPart.Input, toolCallPart.ToolCallID, toolCallPart.ToolName))
					lastEmittedReasoningReference = false
				case fantasy.ContentTypeSource:
					// Source citations from web search are not a
					// recognised Responses API input type; skip.
					continue
				case fantasy.ContentTypeReasoning:
					lastEmittedReasoningReference = false
					if !store {
						// When store is disabled, server-side reasoning
						// items are ephemeral and cannot be referenced.
						continue
					}
					reasoningPart, ok := fantasy.AsContentType[fantasy.ReasoningPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant reasoning part does not have the right type",
						})
						continue
					}
					meta := GetReasoningMetadata(reasoningPart.ProviderOptions)
					if meta == nil || meta.ItemID == "" {
						continue
					}
					input = append(input, responses.ResponseInputItemParamOfItemReference(meta.ItemID))
					lastEmittedReasoningReference = true
					continue
				}
			}
			if !hasVisibleResponsesAssistantContent(input, startIdx) {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "dropping empty assistant message (contains neither user-facing content nor tool calls)",
				})
				// Remove any items that were added during this iteration
				input = input[:startIdx]
				continue
			}

		case fantasy.MessageRoleTool:
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeToolResult {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "tool message can only have tool result content",
					})
					continue
				}

				toolResultPart, ok := fantasy.AsContentType[fantasy.ToolResultPart](c)
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "tool message result part does not have the right type",
					})
					continue
				}

				// Provider-executed tool results (e.g. web search)
				// are already round-tripped via the tool call; skip.
				if toolResultPart.ProviderExecuted {
					continue
				}

				// Computer call outputs require image data and use a
				// dedicated input item type. The tool result is
				// recognized as a computer_call_output when either the
				// originating computer_call is in the prompt history or
				// the caller explicitly attaches
				// ComputerCallOutputOptions (typical when chaining via
				// previous_response_id and replaying only the new turn).
				_, hasComputerCallInPrompt := computerCallRawJSON[toolResultPart.ToolCallID]
				hasComputerCallOutputOptions := GetComputerCallOutputOptions(toolResultPart.ProviderOptions) != nil
				if hasComputerCallInPrompt || hasComputerCallOutputOptions {
					if toolResultPart.Output.GetType() != fantasy.ToolResultContentTypeMedia {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "computer_call_output requires image result; skipping non-image tool result",
						})
						continue
					}
					mediaOutput, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](toolResultPart.Output)
					if !ok || !strings.HasPrefix(mediaOutput.MediaType, "image/") {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "computer_call_output requires image/* media type; skipping",
						})
						continue
					}

					imageURL := fmt.Sprintf("data:%s;base64,%s", mediaOutput.MediaType, mediaOutput.Data)
					outputParam := responses.ResponseInputItemComputerCallOutputParam{
						CallID: toolResultPart.ToolCallID,
						Output: responses.ResponseComputerToolCallOutputScreenshotParam{
							ImageURL: param.NewOpt(imageURL),
						},
					}
					if outputOptions := GetComputerCallOutputOptions(toolResultPart.ProviderOptions); outputOptions != nil && outputOptions.Detail != "" {
						// The pinned openai-go SDK does not yet expose a
						// Detail field on the screenshot param. Inject it
						// via SetExtraFields so the wire payload still
						// carries the documented `detail` key.
						outputParam.Output.SetExtraFields(map[string]any{
							"detail": outputOptions.Detail,
						})
					}

					input = append(input, responses.ResponseInputItemUnionParam{
						OfComputerCallOutput: &outputParam,
					})
					continue
				}

				var outputStr string
				var followupParts responses.ResponseInputMessageContentListParam
				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					outputStr = output.Text
				case fantasy.ToolResultContentTypeError:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					outputStr = output.Error.Error()
				case fantasy.ToolResultContentTypeMedia:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					// The Responses API function_call_output only accepts a
					// string. Emit a text placeholder (preserving any
					// accompanying text) so the tool_call/tool_result pairing
					// stays valid, then attach the media as a synthetic user
					// input_image so vision-capable models still receive it.
					outputStr = output.Text
					if outputStr == "" {
						outputStr = fmt.Sprintf("The tool returned %s content; see the following user message.", output.MediaType)
					}
					if strings.HasPrefix(output.MediaType, "image/") {
						imageURL := fmt.Sprintf("data:%s;base64,%s", output.MediaType, output.Data)
						followupParts = append(followupParts, responses.ResponseInputContentUnionParam{
							OfInputImage: &responses.ResponseInputImageParam{
								Type:     "input_image",
								ImageURL: param.NewOpt(imageURL),
							},
						})
					} else {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: fmt.Sprintf("tool result media type %s not supported, sending text placeholder only", output.MediaType),
						})
					}
				default:
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: fmt.Sprintf("tool result output type %q not supported", toolResultPart.Output.GetType()),
					})
					continue
				}

				input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(toolResultPart.ToolCallID, outputStr))
				if len(followupParts) > 0 {
					input = append(input, responses.ResponseInputItemParamOfMessage(followupParts, responses.EasyInputMessageRoleUser))
				}
			}
		}
	}

	if err := validateResponsesInputWithOptions(input, allowUnpairedFunctionCallOutputs); err != nil {
		return nil, warnings, err
	}

	return input, warnings, nil
}

func isResponsesWebSearchToolCall(toolCallPart fantasy.ToolCallPart) bool {
	return toolCallPart.ToolName == "web_search" ||
		toolCallPart.ToolName == "web_search_preview"
}

func validateResponsesInput(input responses.ResponseInputParam) error {
	if err := validateResponsesFunctionCallOutputs(input); err != nil {
		return err
	}
	return validateResponsesItemReferences(input)
}

func validateResponsesInputWithOptions(input responses.ResponseInputParam, allowUnpairedFunctionCallOutputs bool) error {
	if !allowUnpairedFunctionCallOutputs {
		return validateResponsesInput(input)
	}
	return validateResponsesItemReferences(input)
}

func validateResponsesFunctionCallOutputs(input responses.ResponseInputParam) error {
	type callState struct {
		calls       int
		outputs     int
		firstCall   int
		firstOutput int
	}
	states := make(map[string]*callState)
	var callIDs []string
	var outputIDs []string

	stateFor := func(callID string) *callState {
		state, ok := states[callID]
		if ok {
			return state
		}
		state = &callState{firstCall: -1, firstOutput: -1}
		states[callID] = state
		return state
	}

	for index, item := range input {
		if item.OfFunctionCall != nil {
			callID := item.OfFunctionCall.CallID
			state := stateFor(callID)
			if state.calls == 0 {
				callIDs = append(callIDs, callID)
				state.firstCall = index
			}
			state.calls++
		}

		if item.OfFunctionCallOutput != nil {
			callID := item.OfFunctionCallOutput.CallID
			state := stateFor(callID)
			if state.outputs == 0 {
				outputIDs = append(outputIDs, callID)
				state.firstOutput = index
			}
			state.outputs++
		}
	}

	for _, callID := range callIDs {
		state := states[callID]
		if state.calls > 1 {
			return fmt.Errorf("openai responses prompt has duplicate function_call for call_id %q", callID)
		}
	}
	for _, callID := range outputIDs {
		state := states[callID]
		if state.outputs > 1 {
			return fmt.Errorf("openai responses prompt has duplicate function_call_output for call_id %q", callID)
		}
	}
	for _, callID := range outputIDs {
		state := states[callID]
		if state.calls == 0 {
			return fmt.Errorf("openai responses prompt has function_call_output without function_call for call_id %q", callID)
		}
		if state.firstOutput < state.firstCall {
			return fmt.Errorf("openai responses prompt has function_call_output before function_call for call_id %q", callID)
		}
	}
	for _, callID := range callIDs {
		state := states[callID]
		if state.outputs == 0 {
			return fmt.Errorf("openai responses prompt has function_call without function_call_output for call_id %q", callID)
		}
	}

	return nil
}

func validateResponsesItemReferences(input responses.ResponseInputParam) error {
	previousReferenceID := ""
	for _, item := range input {
		if item.OfItemReference == nil {
			previousReferenceID = ""
			continue
		}

		itemID := item.OfItemReference.ID
		if strings.HasPrefix(itemID, "ws_") && !strings.HasPrefix(previousReferenceID, "rs_") {
			return fmt.Errorf("openai responses prompt has web_search_call item_reference without preceding reasoning item_reference for item_id %q", itemID)
		}
		previousReferenceID = itemID
	}
	return nil
}

func hasVisibleResponsesUserContent(content responses.ResponseInputMessageContentListParam) bool {
	return len(content) > 0
}

func hasVisibleResponsesAssistantContent(items []responses.ResponseInputItemUnionParam, startIdx int) bool {
	// Check if we added any assistant content parts from this message
	for i := startIdx; i < len(items); i++ {
		if items[i].OfMessage != nil || items[i].OfFunctionCall != nil || items[i].OfItemReference != nil || items[i].OfComputerCall != nil {
			return true
		}
	}
	return false
}

func toResponsesTools(tools []fantasy.Tool, toolChoice *fantasy.ToolChoice, options *ResponsesProviderOptions) ([]responses.ToolUnionParam, responses.ResponseNewParamsToolChoiceUnion, []fantasy.CallWarning) {
	warnings := make([]fantasy.CallWarning, 0)
	var openaiTools []responses.ToolUnionParam

	if len(tools) == 0 {
		return nil, responses.ResponseNewParamsToolChoiceUnion{}, nil
	}

	strictJSONSchema := false
	if options != nil && options.StrictJSONSchema != nil {
		strictJSONSchema = *options.StrictJSONSchema
	}

	for _, tool := range tools {
		if tool.GetType() == fantasy.ToolTypeFunction {
			ft, ok := tool.(fantasy.FunctionTool)
			if !ok {
				continue
			}
			openaiTools = append(openaiTools, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        ft.Name,
					Description: param.NewOpt(ft.Description),
					Parameters:  ft.InputSchema,
					Strict:      param.NewOpt(strictJSONSchema),
					Type:        "function",
				},
			})
			continue
		}
		if IsComputerUseTool(tool) {
			computerParam := responses.NewComputerToolParam()
			openaiTools = append(openaiTools, responses.ToolUnionParam{
				OfComputer: &computerParam,
			})
			continue
		}
		if tool.GetType() == fantasy.ToolTypeProviderDefined {
			pt, ok := asProviderDefinedTool(tool)
			if !ok {
				continue
			}
			switch pt.ID {
			case "web_search":
				openaiTools = append(openaiTools, toWebSearchToolParam(pt))
				continue
			}
		}

		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}

	if toolChoice == nil {
		return openaiTools, responses.ResponseNewParamsToolChoiceUnion{}, warnings
	}

	var openaiToolChoice responses.ResponseNewParamsToolChoiceUnion

	switch *toolChoice {
	case fantasy.ToolChoiceAuto:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
		}
	case fantasy.ToolChoiceNone:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
		}
	case fantasy.ToolChoiceRequired:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		}
	default:
		if string(*toolChoice) == "computer" && hasComputerUseTool(tools) {
			openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfHostedTool: &responses.ToolChoiceTypesParam{
					Type: responses.ToolChoiceTypesTypeComputer,
				},
			}
		} else {
			openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Type: "function",
					Name: string(*toolChoice),
				},
			}
		}
	}
	return openaiTools, openaiToolChoice, warnings
}

func (o responsesLanguageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	params, warnings, err := o.prepareParams(call)
	if err != nil {
		return nil, err
	}

	response, err := o.client.Responses.New(ctx, *params, append(callUARequestOptions(call), callHeadersRequestOptions(call)...)...)
	if err != nil {
		return nil, toProviderErr(err)
	}

	if response == nil {
		return nil, &fantasy.Error{Title: "no response", Message: "provider returned nil response"}
	}

	if response.Error.Message != "" {
		return nil, &fantasy.Error{
			Title:   "provider error",
			Message: fmt.Sprintf("%s (code: %s)", response.Error.Message, response.Error.Code),
		}
	}

	var content []fantasy.Content
	hasFunctionCall := false

	for _, outputItem := range response.Output {
		switch outputItem.Type {
		case "message":
			for _, contentPart := range outputItem.Content {
				if contentPart.Type == "output_text" {
					content = append(content, fantasy.TextContent{
						Text: contentPart.Text,
					})

					for _, annotation := range contentPart.Annotations {
						switch annotation.Type {
						case "url_citation":
							content = append(content, fantasy.SourceContent{
								SourceType: fantasy.SourceTypeURL,
								ID:         uuid.NewString(),
								URL:        annotation.URL,
								Title:      annotation.Title,
							})
						case "file_citation":
							title := "Document"
							if annotation.Filename != "" {
								title = annotation.Filename
							}
							filename := annotation.Filename
							if filename == "" {
								filename = annotation.FileID
							}
							content = append(content, fantasy.SourceContent{
								SourceType: fantasy.SourceTypeDocument,
								ID:         uuid.NewString(),
								MediaType:  "text/plain",
								Title:      title,
								Filename:   filename,
							})
						}
					}
				}
			}

		case "function_call":
			hasFunctionCall = true
			content = append(content, fantasy.ToolCallContent{
				ProviderExecuted: false,
				ToolCallID:       outputItem.CallID,
				ToolName:         outputItem.Name,
				Input:            outputItem.Arguments.OfString,
			})

		case "web_search_call":
			// Provider-executed web search tool call. Emit both
			// a ToolCallContent and ToolResultContent as a pair,
			// matching the vercel/ai pattern for provider tools.
			//
			// Note: source citations come from url_citation annotations
			// on the message text (handled in the "message" case above),
			// not from the web_search_call action.
			wsMeta := webSearchCallToMetadata(outputItem.ID, outputItem.Action)
			content = append(content, fantasy.ToolCallContent{
				ProviderExecuted: true,
				ToolCallID:       outputItem.ID,
				ToolName:         "web_search",
			})
			content = append(content, fantasy.ToolResultContent{
				ProviderExecuted: true,
				ToolCallID:       outputItem.ID,
				ToolName:         "web_search",
				ProviderMetadata: fantasy.ProviderMetadata{
					Name: wsMeta,
				},
			})

		case "computer_call":
			computerCall := outputItem.AsComputerCall()
			inputJSON, err := computerCallInput(computerCall)
			if err != nil {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: fmt.Sprintf("malformed computer_call (call_id %q): %v", outputItem.CallID, err),
				})
				continue
			}
			hasFunctionCall = true
			content = append(content, fantasy.ToolCallContent{
				ProviderExecuted: false,
				ToolCallID:       computerCall.CallID,
				ToolName:         "computer",
				Input:            inputJSON,
				ProviderMetadata: fantasy.ProviderMetadata{
					Name: &ComputerUseMetadata{RawJSON: outputItem.RawJSON()},
				},
			})
		case "reasoning":
			metadata := &ResponsesReasoningMetadata{
				ItemID: outputItem.ID,
			}
			if outputItem.EncryptedContent != "" {
				metadata.EncryptedContent = &outputItem.EncryptedContent
			}

			if len(outputItem.Summary) == 0 && metadata.EncryptedContent == nil {
				continue
			}

			// When there are no summary parts, add an empty reasoning part
			summaries := outputItem.Summary
			if len(summaries) == 0 {
				summaries = []responses.ResponseReasoningItemSummary{{Type: "summary_text", Text: ""}}
			}

			for _, s := range summaries {
				metadata.Summary = append(metadata.Summary, s.Text)
			}

			content = append(content, fantasy.ReasoningContent{
				Text: strings.Join(metadata.Summary, "\n"),
				ProviderMetadata: fantasy.ProviderMetadata{
					Name: metadata,
				},
			})
		}
	}

	usage := responsesUsage(*response)
	finishReason := mapResponsesFinishReason(response.IncompleteDetails.Reason, hasFunctionCall)

	return &fantasy.Response{
		Content:          content,
		Usage:            usage,
		FinishReason:     finishReason,
		ProviderMetadata: responsesProviderMetadata(response.ID),
		Warnings:         warnings,
	}, nil
}

func mapResponsesFinishReason(reason string, hasFunctionCall bool) fantasy.FinishReason {
	if hasFunctionCall {
		return fantasy.FinishReasonToolCalls
	}

	switch reason {
	case "":
		return fantasy.FinishReasonStop
	case "max_tokens", "max_output_tokens":
		return fantasy.FinishReasonLength
	case "content_filter":
		return fantasy.FinishReasonContentFilter
	default:
		return fantasy.FinishReasonOther
	}
}

func (o responsesLanguageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	params, warnings, err := o.prepareParams(call)
	if err != nil {
		return nil, err
	}

	stream := o.client.Responses.NewStreaming(ctx, *params, append(callUARequestOptions(call), callHeadersRequestOptions(call)...)...)

	finishReason := fantasy.FinishReasonUnknown
	var usage fantasy.Usage
	// responseID tracks the server-assigned response ID. It's first set from the
	// response.created event and may be overwritten by response.completed or
	// response.incomplete events. Per the OpenAI API contract, these IDs are
	// identical; the overwrites ensure we have the final value even if an event
	// is missed.
	responseID := ""
	sawTerminalEvent := false
	ongoingToolCalls := make(map[int64]*ongoingToolCall)
	hasFunctionCall := false
	activeReasoning := make(map[string]*reasoningState)

	return func(yield func(fantasy.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(fantasy.StreamPart{
				Type:     fantasy.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.created":
				created := event.AsResponseCreated()
				responseID = created.Response.ID

			case "response.output_item.added":
				added := event.AsResponseOutputItemAdded()
				switch added.Item.Type {
				case "function_call":
					ongoingToolCalls[added.OutputIndex] = &ongoingToolCall{
						toolName:   added.Item.Name,
						toolCallID: added.Item.CallID,
					}
					if !yield(fantasy.StreamPart{
						Type:         fantasy.StreamPartTypeToolInputStart,
						ID:           added.Item.CallID,
						ToolCallName: added.Item.Name,
					}) {
						return
					}

				case "web_search_call":
					// Provider-executed web search; emit start.
					if !yield(fantasy.StreamPart{
						Type:             fantasy.StreamPartTypeToolInputStart,
						ID:               added.Item.ID,
						ToolCallName:     "web_search",
						ProviderExecuted: true,
					}) {
						return
					}

				case "computer_call":
					ongoingToolCalls[added.OutputIndex] = &ongoingToolCall{
						toolName:   "computer",
						toolCallID: added.Item.CallID,
					}
					if !yield(fantasy.StreamPart{
						Type:         fantasy.StreamPartTypeToolInputStart,
						ID:           added.Item.CallID,
						ToolCallName: "computer",
					}) {
						return
					}

				case "message":
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextStart,
						ID:   added.Item.ID,
					}) {
						return
					}

				case "reasoning":
					metadata := &ResponsesReasoningMetadata{
						ItemID:  added.Item.ID,
						Summary: []string{},
					}
					if added.Item.EncryptedContent != "" {
						metadata.EncryptedContent = &added.Item.EncryptedContent
					}

					activeReasoning[added.Item.ID] = &reasoningState{
						metadata: metadata,
					}
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningStart,
						ID:   added.Item.ID,
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: metadata,
						},
					}) {
						return
					}
				}

			case "response.output_item.done":
				done := event.AsResponseOutputItemDone()
				switch done.Item.Type {
				case "function_call":
					tc := ongoingToolCalls[done.OutputIndex]
					if tc != nil {
						delete(ongoingToolCalls, done.OutputIndex)
						hasFunctionCall = true

						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeToolInputEnd,
							ID:   done.Item.CallID,
						}) {
							return
						}
						if !yield(fantasy.StreamPart{
							Type:          fantasy.StreamPartTypeToolCall,
							ID:            done.Item.CallID,
							ToolCallName:  done.Item.Name,
							ToolCallInput: done.Item.Arguments.OfString,
						}) {
							return
						}
					}

				case "web_search_call":
					// Provider-executed web search completed.
					// Source citations come from url_citation annotations
					// on the streamed message text, not from the action.
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeToolInputEnd,
						ID:   done.Item.ID,
					}) {
						return
					}
					if !yield(fantasy.StreamPart{
						Type:             fantasy.StreamPartTypeToolCall,
						ID:               done.Item.ID,
						ToolCallName:     "web_search",
						ProviderExecuted: true,
					}) {
						return
					}
					// Emit a ToolResult so the agent framework
					// includes it in round-trip messages.
					if !yield(fantasy.StreamPart{
						Type:             fantasy.StreamPartTypeToolResult,
						ID:               done.Item.ID,
						ToolCallName:     "web_search",
						ProviderExecuted: true,
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: webSearchCallToMetadata(done.Item.ID, done.Item.Action),
						},
					}) {
						return
					}
				case "message":
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextEnd,
						ID:   done.Item.ID,
					}) {
						return
					}

				case "computer_call":
					tc := ongoingToolCalls[done.OutputIndex]
					if tc != nil {
						delete(ongoingToolCalls, done.OutputIndex)
						hasFunctionCall = true
						computerCall := done.Item.AsComputerCall()
						inputJSON, err := computerCallInput(computerCall)
						if err != nil {
							// Malformed computer_call — skip silently.
							// We cannot emit StreamPartTypeError here
							// because the agent treats it as fatal.
							continue
						}
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeToolInputEnd,
							ID:   computerCall.CallID,
						}) {
							return
						}
						if !yield(fantasy.StreamPart{
							Type:          fantasy.StreamPartTypeToolCall,
							ID:            computerCall.CallID,
							ToolCallName:  "computer",
							ToolCallInput: inputJSON,
							ProviderMetadata: fantasy.ProviderMetadata{
								Name: &ComputerUseMetadata{RawJSON: done.Item.RawJSON()},
							},
						}) {
							return
						}
					}
				case "reasoning":
					state := activeReasoning[done.Item.ID]
					if state != nil {
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeReasoningEnd,
							ID:   done.Item.ID,
							ProviderMetadata: fantasy.ProviderMetadata{
								Name: state.metadata,
							},
						}) {
							return
						}
						delete(activeReasoning, done.Item.ID)
					}
				}

			case "response.function_call_arguments.delta":
				delta := event.AsResponseFunctionCallArgumentsDelta()
				tc := ongoingToolCalls[delta.OutputIndex]
				if tc != nil {
					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeToolInputDelta,
						ID:    tc.toolCallID,
						Delta: delta.Delta,
					}) {
						return
					}
				}

			case "response.output_text.delta":
				textDelta := event.AsResponseOutputTextDelta()
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					ID:    textDelta.ItemID,
					Delta: textDelta.Delta,
				}) {
					return
				}

			case "response.output_text.annotation.added":
				added := event.AsResponseOutputTextAnnotationAdded()
				// The Annotation field is typed as `any` in the SDK;
				// it deserializes as map[string]any from JSON.
				annotationMap, ok := added.Annotation.(map[string]any)
				if !ok {
					break
				}
				annotationType, _ := annotationMap["type"].(string)
				switch annotationType {
				case "url_citation":
					url, _ := annotationMap["url"].(string)
					title, _ := annotationMap["title"].(string)
					if !yield(fantasy.StreamPart{
						Type:       fantasy.StreamPartTypeSource,
						ID:         uuid.NewString(),
						SourceType: fantasy.SourceTypeURL,
						URL:        url,
						Title:      title,
					}) {
						return
					}
				case "file_citation":
					title := "Document"
					if fn, ok := annotationMap["filename"].(string); ok && fn != "" {
						title = fn
					}
					if !yield(fantasy.StreamPart{
						Type:       fantasy.StreamPartTypeSource,
						ID:         uuid.NewString(),
						SourceType: fantasy.SourceTypeDocument,
						Title:      title,
					}) {
						return
					}
				}

			case "response.reasoning_summary_part.added":
				added := event.AsResponseReasoningSummaryPartAdded()
				state := activeReasoning[added.ItemID]
				if state != nil {
					state.metadata.Summary = append(state.metadata.Summary, "")
					activeReasoning[added.ItemID] = state
					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeReasoningDelta,
						ID:    added.ItemID,
						Delta: "\n",
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: state.metadata,
						},
					}) {
						return
					}
				}

			case "response.reasoning_summary_text.delta":
				textDelta := event.AsResponseReasoningSummaryTextDelta()
				state := activeReasoning[textDelta.ItemID]
				if state != nil {
					if len(state.metadata.Summary)-1 >= int(textDelta.SummaryIndex) {
						state.metadata.Summary[textDelta.SummaryIndex] += textDelta.Delta
					}
					activeReasoning[textDelta.ItemID] = state
					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeReasoningDelta,
						ID:    textDelta.ItemID,
						Delta: textDelta.Delta,
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: state.metadata,
						},
					}) {
						return
					}
				}

			case "response.completed":
				sawTerminalEvent = true
				completed := event.AsResponseCompleted()
				responseID = completed.Response.ID
				finishReason = mapResponsesFinishReason(completed.Response.IncompleteDetails.Reason, hasFunctionCall)
				usage = responsesUsage(completed.Response)

			case "response.incomplete":
				sawTerminalEvent = true
				incomplete := event.AsResponseIncomplete()
				responseID = incomplete.Response.ID
				finishReason = mapResponsesFinishReason(incomplete.Response.IncompleteDetails.Reason, hasFunctionCall)
				usage = responsesUsage(incomplete.Response)

			case "response.failed":
				failed := event.AsResponseFailed()
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: responsesFailedStreamError(failed.Response.Error.Message, string(failed.Response.Error.Code)),
				}) {
					return
				}
				return

			case "error":
				errorEvent := event.AsError()
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: responsesErrorStreamError(errorEvent.Message, errorEvent.Code),
				}) {
					return
				}
				return
			}
		}

		err := stream.Err()
		if err != nil && !errors.Is(err, io.EOF) {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: toProviderErr(err),
			})
			return
		}
		if !sawTerminalEvent {
			err := ctx.Err()
			if err == nil {
				err = fantasy.NewIncompleteStreamError()
			}
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: err,
			})
			return
		}

		yield(fantasy.StreamPart{
			Type:             fantasy.StreamPartTypeFinish,
			Usage:            usage,
			FinishReason:     finishReason,
			ProviderMetadata: responsesProviderMetadata(responseID),
		})
	}, nil
}

// responsesFailedStreamError intentionally returns a provider-declared failure
// instead of a retryable transport error. Only synthetic stream truncation
// errors are wrapped with io.ErrUnexpectedEOF.
func responsesFailedStreamError(message, code string) error {
	return responsesStreamFailureError("response failed", message, code)
}

func responsesErrorStreamError(message, code string) error {
	return responsesStreamFailureError("response error", message, code)
}

func responsesStreamFailureError(title, message, code string) error {
	if code != "" {
		message = fmt.Sprintf("%s (code: %s)", message, code)
	}
	return &fantasy.Error{Title: title, Message: message}
}

// toWebSearchToolParam converts a ProviderDefinedTool with ID
// "web_search" into the OpenAI SDK's WebSearchToolParam.
func toWebSearchToolParam(pt fantasy.ProviderDefinedTool) responses.ToolUnionParam {
	wst := responses.WebSearchToolParam{
		Type: responses.WebSearchToolTypeWebSearch,
	}
	if pt.Args != nil {
		if size, ok := pt.Args["search_context_size"].(SearchContextSize); ok && size != "" {
			wst.SearchContextSize = responses.WebSearchToolSearchContextSize(size)
		}
		// Also accept plain string for search_context_size.
		if size, ok := pt.Args["search_context_size"].(string); ok && size != "" {
			wst.SearchContextSize = responses.WebSearchToolSearchContextSize(size)
		}
		if domains, ok := pt.Args["allowed_domains"].([]string); ok && len(domains) > 0 {
			wst.Filters.AllowedDomains = domains
		}
		if loc, ok := pt.Args["user_location"].(*WebSearchUserLocation); ok && loc != nil {
			if loc.City != "" {
				wst.UserLocation.City = param.NewOpt(loc.City)
			}
			if loc.Region != "" {
				wst.UserLocation.Region = param.NewOpt(loc.Region)
			}
			if loc.Country != "" {
				wst.UserLocation.Country = param.NewOpt(loc.Country)
			}
			if loc.Timezone != "" {
				wst.UserLocation.Timezone = param.NewOpt(loc.Timezone)
			}
		}
	}
	return responses.ToolUnionParam{
		OfWebSearch: &wst,
	}
}

// webSearchCallToMetadata converts an OpenAI web search call output
// into our structured metadata for round-tripping.
func webSearchCallToMetadata(itemID string, action responses.ResponseOutputItemUnionAction) *WebSearchCallMetadata {
	meta := &WebSearchCallMetadata{ItemID: itemID}
	if action.Type != "" {
		a := &WebSearchAction{
			Type:  action.Type,
			Query: action.Query,
		}
		for _, src := range action.Sources {
			a.Sources = append(a.Sources, WebSearchSource{
				Type: string(src.Type),
				URL:  src.URL,
			})
		}
		meta.Action = a
	}
	return meta
}

// GetReasoningMetadata extracts reasoning metadata from provider options for responses models.
func GetReasoningMetadata(providerOptions fantasy.ProviderOptions) *ResponsesReasoningMetadata {
	if openaiResponsesOptions, ok := providerOptions[Name]; ok {
		if reasoning, ok := openaiResponsesOptions.(*ResponsesReasoningMetadata); ok {
			return reasoning
		}
	}
	return nil
}

type ongoingToolCall struct {
	toolName   string
	toolCallID string
}

// hasComputerUseTool reports whether any tool in the list is a
// computer use tool.
func hasComputerUseTool(tools []fantasy.Tool) bool {
	return slices.ContainsFunc(tools, IsComputerUseTool)
}

type reasoningState struct {
	metadata *ResponsesReasoningMetadata
}

// GenerateObject implements fantasy.LanguageModel.
func (o responsesLanguageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	switch o.objectMode {
	case fantasy.ObjectModeText:
		return object.GenerateWithText(ctx, o, call)
	case fantasy.ObjectModeTool:
		return object.GenerateWithTool(ctx, o, call)
	default:
		return o.generateObjectWithJSONMode(ctx, call)
	}
}

// StreamObject implements fantasy.LanguageModel.
func (o responsesLanguageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	switch o.objectMode {
	case fantasy.ObjectModeTool:
		return object.StreamWithTool(ctx, o, call)
	case fantasy.ObjectModeText:
		return object.StreamWithText(ctx, o, call)
	default:
		return o.streamObjectWithJSONMode(ctx, call)
	}
}

func (o responsesLanguageModel) generateObjectWithJSONMode(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	// Convert our Schema to OpenAI's JSON Schema format
	jsonSchemaMap := schema.ToMap(call.Schema)

	// Add additionalProperties: false recursively for strict mode (OpenAI requirement)
	addAdditionalPropertiesFalse(jsonSchemaMap)

	schemaName := call.SchemaName
	if schemaName == "" {
		schemaName = "response"
	}

	// Build request using prepareParams
	fantasyCall := fantasy.Call{
		Prompt:           call.Prompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	}

	params, warnings, err := o.prepareParams(fantasyCall)
	if err != nil {
		return nil, err
	}

	// Add structured output via Text.Format field
	params.Text = responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigParamOfJSONSchema(schemaName, jsonSchemaMap),
	}

	// Make request
	response, err := o.client.Responses.New(ctx, *params, append(objectCallUARequestOptions(call), objectCallHeadersRequestOptions(call)...)...)
	if err != nil {
		return nil, toProviderErr(err)
	}

	if response.Error.Message != "" {
		return nil, &fantasy.Error{
			Title:   "provider error",
			Message: fmt.Sprintf("%s (code: %s)", response.Error.Message, response.Error.Code),
		}
	}

	// Extract JSON text from response
	var jsonText string
	for _, outputItem := range response.Output {
		if outputItem.Type == "message" {
			for _, contentPart := range outputItem.Content {
				if contentPart.Type == "output_text" {
					jsonText = contentPart.Text
					break
				}
			}
		}
	}

	if jsonText == "" {
		usage := fantasy.Usage{
			InputTokens:  response.Usage.InputTokens,
			OutputTokens: response.Usage.OutputTokens,
			TotalTokens:  response.Usage.InputTokens + response.Usage.OutputTokens,
		}
		finishReason := mapResponsesFinishReason(response.IncompleteDetails.Reason, false)
		return nil, &fantasy.NoObjectGeneratedError{
			RawText:      "",
			ParseError:   fmt.Errorf("no text content in response"),
			Usage:        usage,
			FinishReason: finishReason,
		}
	}

	// Parse and validate
	var obj any
	if call.RepairText != nil {
		obj, err = schema.ParseAndValidateWithRepair(ctx, jsonText, call.Schema, call.RepairText)
	} else {
		obj, err = schema.ParseAndValidate(jsonText, call.Schema)
	}

	usage := responsesUsage(*response)
	finishReason := mapResponsesFinishReason(response.IncompleteDetails.Reason, false)

	if err != nil {
		// Add usage info to error
		if nogErr, ok := err.(*fantasy.NoObjectGeneratedError); ok {
			nogErr.Usage = usage
			nogErr.FinishReason = finishReason
		}
		return nil, err
	}

	return &fantasy.ObjectResponse{
		Object:           obj,
		RawText:          jsonText,
		Usage:            usage,
		FinishReason:     finishReason,
		Warnings:         warnings,
		ProviderMetadata: responsesProviderMetadata(response.ID),
	}, nil
}

func (o responsesLanguageModel) streamObjectWithJSONMode(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	// Convert our Schema to OpenAI's JSON Schema format
	jsonSchemaMap := schema.ToMap(call.Schema)

	// Add additionalProperties: false recursively for strict mode (OpenAI requirement)
	addAdditionalPropertiesFalse(jsonSchemaMap)

	schemaName := call.SchemaName
	if schemaName == "" {
		schemaName = "response"
	}

	// Build request using prepareParams
	fantasyCall := fantasy.Call{
		Prompt:           call.Prompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	}

	params, warnings, err := o.prepareParams(fantasyCall)
	if err != nil {
		return nil, err
	}

	// Add structured output via Text.Format field
	params.Text = responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigParamOfJSONSchema(schemaName, jsonSchemaMap),
	}

	stream := o.client.Responses.NewStreaming(ctx, *params, append(objectCallUARequestOptions(call), objectCallHeadersRequestOptions(call)...)...)

	return func(yield func(fantasy.ObjectStreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(fantasy.ObjectStreamPart{
				Type:     fantasy.ObjectStreamPartTypeObject,
				Warnings: warnings,
			}) {
				return
			}
		}

		var accumulated string
		var lastParsedObject any
		var usage fantasy.Usage
		var finishReason fantasy.FinishReason
		// responseID tracks the server-assigned response ID. It's first set from the
		// response.created event and may be overwritten by response.completed or
		// response.incomplete events. Per the OpenAI API contract, these IDs are
		// identical; the overwrites ensure we have the final value even if an event
		// is missed.
		var responseID string
		var sawTerminalEvent bool
		hasFunctionCall := false

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.created":
				created := event.AsResponseCreated()
				responseID = created.Response.ID

			case "response.output_text.delta":
				textDelta := event.AsResponseOutputTextDelta()
				accumulated += textDelta.Delta

				// Try to parse the accumulated text
				obj, state, parseErr := schema.ParsePartialJSON(accumulated)

				// If we successfully parsed, validate and emit
				if state == schema.ParseStateSuccessful || state == schema.ParseStateRepaired {
					if err := schema.ValidateAgainstSchema(obj, call.Schema); err == nil {
						// Only emit if object is different from last
						if !reflect.DeepEqual(obj, lastParsedObject) {
							if !yield(fantasy.ObjectStreamPart{
								Type:   fantasy.ObjectStreamPartTypeObject,
								Object: obj,
							}) {
								return
							}
							lastParsedObject = obj
						}
					}
				}

				// If parsing failed and we have a repair function, try it
				if state == schema.ParseStateFailed && call.RepairText != nil {
					repairedText, repairErr := call.RepairText(ctx, accumulated, parseErr)
					if repairErr == nil {
						obj2, state2, _ := schema.ParsePartialJSON(repairedText)
						if (state2 == schema.ParseStateSuccessful || state2 == schema.ParseStateRepaired) &&
							schema.ValidateAgainstSchema(obj2, call.Schema) == nil {
							if !reflect.DeepEqual(obj2, lastParsedObject) {
								if !yield(fantasy.ObjectStreamPart{
									Type:   fantasy.ObjectStreamPartTypeObject,
									Object: obj2,
								}) {
									return
								}
								lastParsedObject = obj2
							}
						}
					}
				}

			case "response.completed":
				sawTerminalEvent = true
				completed := event.AsResponseCompleted()
				responseID = completed.Response.ID
				finishReason = mapResponsesFinishReason(completed.Response.IncompleteDetails.Reason, hasFunctionCall)
				usage = responsesUsage(completed.Response)

			case "response.incomplete":
				sawTerminalEvent = true
				incomplete := event.AsResponseIncomplete()
				responseID = incomplete.Response.ID
				finishReason = mapResponsesFinishReason(incomplete.Response.IncompleteDetails.Reason, hasFunctionCall)
				usage = responsesUsage(incomplete.Response)

			case "response.failed":
				failed := event.AsResponseFailed()
				if !yield(fantasy.ObjectStreamPart{
					Type:  fantasy.ObjectStreamPartTypeError,
					Error: responsesFailedStreamError(failed.Response.Error.Message, string(failed.Response.Error.Code)),
				}) {
					return
				}
				return

			case "error":
				errorEvent := event.AsError()
				if !yield(fantasy.ObjectStreamPart{
					Type:  fantasy.ObjectStreamPartTypeError,
					Error: responsesErrorStreamError(errorEvent.Message, errorEvent.Code),
				}) {
					return
				}
				return
			}
		}

		err := stream.Err()
		if err != nil && !errors.Is(err, io.EOF) {
			yield(fantasy.ObjectStreamPart{
				Type:  fantasy.ObjectStreamPartTypeError,
				Error: toProviderErr(err),
			})
			return
		}
		if !sawTerminalEvent {
			err := ctx.Err()
			if err == nil {
				err = fantasy.NewIncompleteStreamError()
			}
			yield(fantasy.ObjectStreamPart{
				Type:  fantasy.ObjectStreamPartTypeError,
				Error: err,
			})
			return
		}

		// Final validation and emit
		if lastParsedObject != nil {
			yield(fantasy.ObjectStreamPart{
				Type:             fantasy.ObjectStreamPartTypeFinish,
				Usage:            usage,
				FinishReason:     finishReason,
				ProviderMetadata: responsesProviderMetadata(responseID),
			})
		} else {
			// No object was generated
			yield(fantasy.ObjectStreamPart{
				Type: fantasy.ObjectStreamPartTypeError,
				Error: &fantasy.NoObjectGeneratedError{
					RawText:      accumulated,
					ParseError:   fmt.Errorf("no valid object generated in stream"),
					Usage:        usage,
					FinishReason: finishReason,
				},
			})
		}
	}, nil
}
