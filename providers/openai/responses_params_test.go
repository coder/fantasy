package openai

import (
	"context"
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"
)

func TestPrepareParams_ReasoningModelClassification(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		modelID       string
		force         *bool
		wantReasoning bool
		wantRole      string
	}{
		{name: "gpt-5", modelID: "gpt-5", wantReasoning: true, wantRole: "developer"},
		{name: "gpt-6", modelID: "gpt-6-astra", wantReasoning: true, wantRole: "developer"},
		{name: "uppercase", modelID: "GPT-6-ASTRA", wantReasoning: true, wantRole: "developer"},
		{name: "gpt-10", modelID: "gpt-10-turbo", wantReasoning: true, wantRole: "developer"},
		{name: "gpt-40", modelID: "gpt-40-turbo", wantReasoning: true, wantRole: "developer"},
		{name: "gpt-100", modelID: "gpt-100-turbo", wantReasoning: true, wantRole: "developer"},
		{name: "gpt-5-chat", modelID: "gpt-5-chat-latest", wantRole: "system"},
		{name: "gpt-6-chat", modelID: "gpt-6-chat-latest", wantRole: "system"},
		{name: "gpt-4o", modelID: "gpt-4o", wantRole: "system"},
		{name: "unknown default", modelID: "totally-new-model", wantRole: "system"},
		{name: "force reasoning", modelID: "totally-new-model", force: new(true), wantReasoning: true, wantRole: "developer"},
		{name: "force non-reasoning", modelID: "gpt-5", force: new(false), wantRole: "system"},
		{name: "preserve remove mode", modelID: "o1-mini", force: new(true), wantReasoning: true},
		{name: "override remove mode", modelID: "o1-mini", force: new(false), wantRole: "system"},
		{name: "agree non-reasoning", modelID: "gpt-4o", force: new(false), wantRole: "system"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			opts := []Option{WithUseResponsesAPI()}
			if tt.modelID == "totally-new-model" || tt.modelID == "o1-mini" {
				opts = append(opts, WithResponsesAPIFunc(func(string) bool { return true }))
			}
			if tt.force != nil {
				opts = append(opts, WithReasoningModelFunc(func(modelID string) bool {
					require.Equal(t, tt.modelID, modelID)
					return *tt.force
				}))
			}
			provider, err := New(opts...)
			require.NoError(t, err)
			model, err := provider.LanguageModel(context.Background(), tt.modelID)
			require.NoError(t, err)
			lm, ok := model.(responsesLanguageModel)
			require.True(t, ok)

			call := testCall(fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleSystem, "Be helpful."),
				testTextMessage(fantasy.MessageRoleUser, "hello"),
			}, &ResponsesProviderOptions{
				ReasoningEffort:  new(ReasoningEffortHigh),
				ReasoningSummary: new("detailed"),
			})
			call.Temperature = new(0.7)
			call.TopP = new(0.9)
			params, warnings, err := lm.prepareParams(call)
			require.NoError(t, err)

			var unsupported []string
			for _, warning := range warnings {
				if warning.Type == fantasy.CallWarningTypeUnsupportedSetting {
					unsupported = append(unsupported, warning.Setting)
				}
			}
			if tt.wantReasoning {
				require.Equal(t, "high", string(params.Reasoning.Effort))
				require.Equal(t, "detailed", string(params.Reasoning.Summary))
				require.False(t, params.Temperature.Valid())
				require.False(t, params.TopP.Valid())
				require.ElementsMatch(t, []string{"temperature", "topP"}, unsupported)
			} else {
				require.Empty(t, params.Reasoning.Effort)
				require.Empty(t, params.Reasoning.Summary)
				require.True(t, params.Temperature.Valid())
				require.Equal(t, 0.7, params.Temperature.Value)
				require.True(t, params.TopP.Valid())
				require.Equal(t, 0.9, params.TopP.Value)
				require.ElementsMatch(t, []string{"reasoningEffort", "reasoningSummary"}, unsupported)
			}

			encoded, err := json.Marshal(params)
			require.NoError(t, err)
			var body map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(encoded, &body))
			_, hasReasoning := body["reasoning"]
			require.Equal(t, tt.wantReasoning, hasReasoning)
			var input []struct {
				Role string `json:"role"`
			}
			require.NoError(t, json.Unmarshal(body["input"], &input))
			if tt.wantRole == "" {
				require.Len(t, input, 1)
				require.Equal(t, "user", input[0].Role)
			} else {
				require.Len(t, input, 2)
				require.Equal(t, tt.wantRole, input[0].Role)
			}
		})
	}
}

func TestPrepareParams_Store(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	prompt := fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}

	tests := []struct {
		name      string
		opts      *ResponsesProviderOptions
		wantStore bool
	}{
		{
			name:      "store true",
			opts:      &ResponsesProviderOptions{Store: new(true)},
			wantStore: true,
		},
		{
			name:      "store false",
			opts:      &ResponsesProviderOptions{Store: new(false)},
			wantStore: false,
		},
		{
			name:      "store default",
			opts:      &ResponsesProviderOptions{},
			wantStore: false,
		},
		{
			name:      "no provider options",
			opts:      nil,
			wantStore: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			params, warnings, err := lm.prepareParams(testCall(prompt, tt.opts))
			require.NoError(t, err)
			require.Empty(t, warnings)
			require.True(t, params.Store.Valid())
			require.Equal(t, tt.wantStore, params.Store.Value)
		})
	}
}

func TestPrepareParams_PreviousResponseID(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	prompt := fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}

	t.Run("forwarded", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(prompt, &ResponsesProviderOptions{
			PreviousResponseID: new("resp_abc123"),
			Store:              new(true),
		}))
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.True(t, params.PreviousResponseID.Valid())
		require.Equal(t, "resp_abc123", params.PreviousResponseID.Value)
	})

	t.Run("not set", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(prompt, &ResponsesProviderOptions{}))
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.False(t, params.PreviousResponseID.Valid())
	})

	t.Run("empty string ignored", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(prompt, &ResponsesProviderOptions{
			PreviousResponseID: new(""),
		}))
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.False(t, params.PreviousResponseID.Valid())
	})
}

func TestPrepareParams_PreviousResponseID_Validation(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	opts := &ResponsesProviderOptions{
		PreviousResponseID: new("resp_abc123"),
		Store:              new(true),
	}

	t.Run("rejects with assistant messages", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
			testTextMessage(fantasy.MessageRoleAssistant, "hi there"),
		}, opts))
		require.EqualError(t, err, previousResponseIDHistoryError)
	})

	t.Run("allows user-only prompt", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
			testTextMessage(fantasy.MessageRoleUser, "follow up"),
		}, opts))
		require.NoError(t, err)
		require.Empty(t, warnings)
	})

	t.Run("allows system + user prompt", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleSystem, "be concise"),
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, opts))
		require.NoError(t, err)
		require.Empty(t, warnings)
	})

	t.Run("allows tool messages", func(t *testing.T) {
		t.Parallel()

		// The OpenAI Responses API supports computer_call_output
		// (and function call_output) items as the next-turn input
		// alongside previous_response_id.
		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testToolResultMessage("done"),
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, opts))
		require.NoError(t, err)
		_ = warnings
	})

	t.Run("rejects without store", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, &ResponsesProviderOptions{
			PreviousResponseID: new("resp_abc123"),
		}))
		require.EqualError(t, err, previousResponseIDStoreError)
	})

	t.Run("rejects with store false", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, &ResponsesProviderOptions{
			PreviousResponseID: new("resp_abc123"),
			Store:              new(false),
		}))
		require.EqualError(t, err, previousResponseIDStoreError)
	})
}

func TestValidatePreviousResponseIDPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		prompt  fantasy.Prompt
		wantErr bool
	}{
		{
			name:   "empty prompt",
			prompt: nil,
		},
		{
			name: "user-only messages",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleUser, "hello"),
				testTextMessage(fantasy.MessageRoleUser, "follow up"),
			},
		},
		{
			name: "system + user messages",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleSystem, "be concise"),
				testTextMessage(fantasy.MessageRoleUser, "hello"),
			},
		},
		{
			name: "contains assistant message",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleAssistant, "hi there"),
			},
			wantErr: true,
		},
		{
			name: "assistant in the middle",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleUser, "hello"),
				testTextMessage(fantasy.MessageRoleAssistant, "hi there"),
				testTextMessage(fantasy.MessageRoleUser, "follow up"),
			},
			wantErr: true,
		},
		{
			name: "tool message before user",
			prompt: fantasy.Prompt{
				testToolResultMessage("done"),
				testTextMessage(fantasy.MessageRoleUser, "follow up"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := validatePreviousResponseIDPrompt(tt.prompt)
			if tt.wantErr {
				require.EqualError(t, err, previousResponseIDHistoryError)
				return
			}

			require.NoError(t, err)
		})
	}
}

func TestResponsesModelAllowlist_GPT56Family(t *testing.T) {
	t.Parallel()

	for _, modelID := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} {
		require.True(t, IsResponsesModel(modelID), modelID)
		require.True(t, IsResponsesReasoningModel(modelID), modelID)
	}
}

func TestResponsesProviderMetadata_Helper(t *testing.T) {
	t.Parallel()

	t.Run("non-empty id", func(t *testing.T) {
		t.Parallel()

		metadata := responsesProviderMetadata("resp_123")
		require.Len(t, metadata, 1)

		providerMetadata, ok := metadata[Name].(*ResponsesProviderMetadata)
		require.True(t, ok)
		require.Equal(t, "resp_123", providerMetadata.ResponseID)
	})

	t.Run("empty id", func(t *testing.T) {
		t.Parallel()

		metadata := responsesProviderMetadata("")
		require.Empty(t, metadata)
	})
}

func TestResponsesProviderMetadata_JSON(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(ResponsesProviderMetadata{ResponseID: "resp_123"})
	require.NoError(t, err)
	require.Contains(t, string(encoded), `"response_id":"resp_123"`)

	decoded, err := fantasy.UnmarshalProviderMetadata(map[string]json.RawMessage{
		Name: encoded,
	})
	require.NoError(t, err)

	providerMetadata, ok := decoded[Name].(*ResponsesProviderMetadata)
	require.True(t, ok)
	require.Equal(t, "resp_123", providerMetadata.ResponseID)
}

func TestPrepareParams_SkipsProviderExecutedToolReferences(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	prompt := fantasy.Prompt{
		testTextMessage(fantasy.MessageRoleUser, "Search for the latest AI news"),
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{
					ToolCallID:       "ws_01",
					ToolName:         "web_search",
					ProviderExecuted: true,
				},
				fantasy.TextPart{Text: "Here is what I found."},
			},
		},
	}

	tests := []struct {
		name string
		opts *ResponsesProviderOptions
	}{
		{
			name: "store true",
			opts: &ResponsesProviderOptions{Store: new(true)},
		},
		{
			name: "store false",
			opts: &ResponsesProviderOptions{Store: new(false)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			params, warnings, err := lm.prepareParams(testCall(prompt, tt.opts))
			require.NoError(t, err)
			require.Empty(t, warnings)

			input := params.Input.OfInputItemList
			require.Len(t, input, 2)
			require.NotNil(t, input[1].OfMessage)
			for _, item := range input {
				require.Nil(t, item.OfItemReference)
				require.Nil(t, item.OfWebSearchCall)
			}

			encoded, err := json.Marshal(params)
			require.NoError(t, err)
			require.Contains(t, string(encoded), "Here is what I found.")
			require.NotContains(t, string(encoded), "ws_01")
			require.NotContains(t, string(encoded), "item_reference")
			require.NotContains(t, string(encoded), "web_search_call")

			items := responseInputItemsFromJSON(t, encoded)
			require.Len(t, items, 2)
			for _, item := range items {
				require.NotEqual(t, "item_reference", item["type"])
				require.NotEqual(t, "web_search_call", item["type"])
				require.NotEqual(t, "ws_01", item["id"])
			}
		})
	}
}

func TestPrepareParams_ValidatesFunctionCallOutputPairing(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()

	t.Run("matching local call and output", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "weather"),
			testResponsesToolCallMessage("call_local"),
			testResponsesToolResultMessage("call_local", "sunny"),
		}, nil))
		require.NoError(t, err)
		require.Empty(t, warnings)

		var functionCalls int
		var functionCallOutputs int
		for _, item := range params.Input.OfInputItemList {
			if item.OfFunctionCall != nil {
				functionCalls++
				require.Equal(t, "call_local", item.OfFunctionCall.CallID)
			}
			if item.OfFunctionCallOutput != nil {
				functionCallOutputs++
				require.Equal(t, "call_local", item.OfFunctionCallOutput.CallID)
			}
		}
		require.Equal(t, 1, functionCalls)
		require.Equal(t, 1, functionCallOutputs)

		encoded, err := json.Marshal(params)
		require.NoError(t, err)
		items := responseInputItemsFromJSON(t, encoded)
		var jsonFunctionCalls int
		var jsonFunctionCallOutputs int
		for _, item := range items {
			switch item["type"] {
			case "function_call":
				jsonFunctionCalls++
				require.Equal(t, "call_local", item["call_id"])
			case "function_call_output":
				jsonFunctionCallOutputs++
				require.Equal(t, "call_local", item["call_id"])
			}
		}
		require.Equal(t, 1, jsonFunctionCalls)
		require.Equal(t, 1, jsonFunctionCallOutputs)
	})

	t.Run("missing local output", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "weather"),
			testResponsesToolCallMessage("call_missing"),
		}, nil))
		require.EqualError(t, err, `openai responses prompt has function_call without function_call_output for call_id "call_missing"`)
		require.Empty(t, warnings)
	})

	t.Run("duplicate local outputs", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testResponsesToolCallMessage("call_duplicate"),
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_duplicate",
						Output:     fantasy.ToolResultOutputContentText{Text: "first"},
					},
					fantasy.ToolResultPart{
						ToolCallID: "call_duplicate",
						Output:     fantasy.ToolResultOutputContentText{Text: "second"},
					},
				},
			},
		}, nil))
		require.EqualError(t, err, `openai responses prompt has duplicate function_call_output for call_id "call_duplicate"`)
		require.Empty(t, warnings)
	})

	t.Run("output without local call", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testResponsesToolResultMessage("call_orphan", "done"),
		}, nil))
		require.EqualError(t, err, `openai responses prompt has function_call_output without function_call for call_id "call_orphan"`)
		require.Empty(t, warnings)
	})

	t.Run("output before local call", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testResponsesToolResultMessage("call_late", "done"),
			testResponsesToolCallMessage("call_late"),
		}, nil))
		require.EqualError(t, err, `openai responses prompt has function_call_output before function_call for call_id "call_late"`)
		require.Empty(t, warnings)
	})

	t.Run("duplicate local calls", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testResponsesToolCallMessage("call_duplicate"),
			testResponsesToolCallMessage("call_duplicate"),
			testResponsesToolResultMessage("call_duplicate", "done"),
		}, nil))
		require.EqualError(t, err, `openai responses prompt has duplicate function_call for call_id "call_duplicate"`)
		require.Empty(t, warnings)
	})

	t.Run("provider executed output is skipped", func(t *testing.T) {
		t.Parallel()

		input, warnings, err := toResponsesPrompt(fantasy.Prompt{
			testResponsesProviderToolResultMessage("ws_01"),
		}, "system", false)
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.Empty(t, input)
	})

	t.Run("provider executed output does not satisfy local call", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testResponsesToolCallMessage("call_provider_result"),
			testResponsesProviderToolResultMessage("call_provider_result"),
		}, nil))
		require.EqualError(t, err, `openai responses prompt has function_call without function_call_output for call_id "call_provider_result"`)
		require.Empty(t, warnings)
	})
}

func TestValidateResponsesInput_WebSearchReferenceRequiresReasoning(t *testing.T) {
	t.Parallel()

	t.Run("valid reasoning and web search references", func(t *testing.T) {
		t.Parallel()

		err := validateResponsesInput(responses.ResponseInputParam{
			responses.ResponseInputItemParamOfItemReference("rs_valid"),
			responses.ResponseInputItemParamOfItemReference("ws_valid"),
		})
		require.NoError(t, err)
	})

	t.Run("web search reference without reasoning", func(t *testing.T) {
		t.Parallel()

		err := validateResponsesInput(responses.ResponseInputParam{
			responses.ResponseInputItemParamOfItemReference("ws_orphan"),
		})
		require.EqualError(t, err, `openai responses prompt has web_search_call item_reference without preceding reasoning item_reference for item_id "ws_orphan"`)
	})

	t.Run("web search reference after non-reference item", func(t *testing.T) {
		t.Parallel()

		err := validateResponsesInput(responses.ResponseInputParam{
			responses.ResponseInputItemParamOfItemReference("rs_valid"),
			responses.ResponseInputItemParamOfMessage("text", responses.EasyInputMessageRoleAssistant),
			responses.ResponseInputItemParamOfItemReference("ws_orphan"),
		})
		require.EqualError(t, err, `openai responses prompt has web_search_call item_reference without preceding reasoning item_reference for item_id "ws_orphan"`)
	})
}

func responseInputItemsFromJSON(t *testing.T, encoded []byte) []map[string]any {
	t.Helper()

	var body map[string]any
	require.NoError(t, json.Unmarshal(encoded, &body))

	rawInput, ok := body["input"].([]any)
	require.True(t, ok)

	items := make([]map[string]any, 0, len(rawInput))
	for _, rawItem := range rawInput {
		item, ok := rawItem.(map[string]any)
		require.True(t, ok)
		items = append(items, item)
	}
	return items
}

func testResponsesToolCallMessage(callID string) fantasy.Message {
	return fantasy.Message{
		Role: fantasy.MessageRoleAssistant,
		Content: []fantasy.MessagePart{
			fantasy.ToolCallPart{
				ToolCallID: callID,
				ToolName:   "get_weather",
				Input:      "{\"location\":\"NYC\"}",
			},
		},
	}
}

func testResponsesToolResultMessage(callID string, text string) fantasy.Message {
	return fantasy.Message{
		Role: fantasy.MessageRoleTool,
		Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{
				ToolCallID: callID,
				Output: fantasy.ToolResultOutputContentText{
					Text: text,
				},
			},
		},
	}
}

func testResponsesProviderToolResultMessage(callID string) fantasy.Message {
	return fantasy.Message{
		Role: fantasy.MessageRoleTool,
		Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{
				ToolCallID:       callID,
				ProviderExecuted: true,
				Output: fantasy.ToolResultOutputContentText{
					Text: "provider result",
				},
			},
		},
	}
}

func testCall(prompt fantasy.Prompt, opts *ResponsesProviderOptions) fantasy.Call {
	call := fantasy.Call{
		Prompt: prompt,
	}
	if opts != nil {
		call.ProviderOptions = fantasy.ProviderOptions{
			Name: opts,
		}
	}
	return call
}

func testResponsesLM() responsesLanguageModel {
	return responsesLanguageModel{
		provider: Name,
		modelID:  "gpt-4o",
	}
}

func testTextMessage(role fantasy.MessageRole, text string) fantasy.Message {
	return fantasy.Message{
		Role: role,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: text},
		},
	}
}

func testToolResultMessage(text string) fantasy.Message {
	return fantasy.Message{
		Role: fantasy.MessageRoleTool,
		Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{
				ToolCallID: "call_123",
				Output: fantasy.ToolResultOutputContentText{
					Text: text,
				},
			},
		},
	}
}
