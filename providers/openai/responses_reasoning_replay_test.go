package openai

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesStream_ReasoningEndUsesCompletedItem(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name        string
		summaryJSON string
		deltas      []string
		wantSummary []string
	}{
		{
			name:        "summary present",
			summaryJSON: `[{"type":"summary_text","text":"final summary"}]`,
			deltas: []string{
				responsesSSEEvent("response.reasoning_summary_part.added", `{"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}`),
				responsesSSEEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"final summary"}`),
			},
			wantSummary: []string{"final summary"},
		},
		{
			name:        "empty summary",
			summaryJSON: `[]`,
			wantSummary: []string{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			sms := newStreamingMockServer()
			defer sms.close()
			sms.chunks = append([]string{
				responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"enc-added","summary":[]}}`),
			}, tc.deltas...)
			sms.chunks = append(sms.chunks,
				responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"enc-final","summary":`+tc.summaryJSON+`}}`),
				responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
			)

			stream, err := newResponsesProvider(t, sms.server.URL).Stream(context.Background(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)

			var start, end *ResponsesReasoningMetadata
			for part := range stream {
				require.NotEqual(t, fantasy.StreamPartTypeError, part.Type, "unexpected stream error: %v", part.Error)
				switch part.Type {
				case fantasy.StreamPartTypeReasoningStart:
					start = GetReasoningMetadata(fantasy.ProviderOptions(part.ProviderMetadata))
				case fantasy.StreamPartTypeReasoningEnd:
					end = GetReasoningMetadata(fantasy.ProviderOptions(part.ProviderMetadata))
				}
			}

			require.NotNil(t, start)
			require.False(t, start.Finalized, "added item metadata must not be replayable")
			require.NotNil(t, end)
			require.NotNil(t, end.EncryptedContent)
			require.Equal(t, "enc-final", *end.EncryptedContent)
			require.Equal(t, &ResponsesReasoningMetadata{
				ItemID:           "rs_1",
				EncryptedContent: new("enc-final"),
				Summary:          tc.wantSummary,
				Finalized:        true,
			}, end)
		})
	}
}

func TestResponsesGenerate_ReasoningMetadataIsFinalized(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id":     "resp_1",
		"object": "response",
		"model":  "gpt-4.1",
		"status": "completed",
		"output": []any{
			map[string]any{"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-final", "summary": []any{}},
			map[string]any{
				"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
				"content": []any{map[string]any{"type": "output_text", "text": "done"}},
			},
		},
		"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
	}

	resp, err := newResponsesProvider(t, server.server.URL).Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)

	reasoning := resp.Content.Reasoning()
	require.Len(t, reasoning, 1)
	meta := GetReasoningMetadata(fantasy.ProviderOptions(reasoning[0].ProviderMetadata))
	require.NotNil(t, meta)
	require.True(t, meta.Finalized)
	require.Equal(t, "enc-final", *meta.EncryptedContent)
}

func TestResponsesReplay_FinalizedReasoningInput(t *testing.T) {
	t.Parallel()

	// The SDK marshals item references without a type field.
	reference := map[string]any{"id": "rs_1"}
	fullItem := func(summary ...any) map[string]any {
		return map[string]any{
			"type":              "reasoning",
			"id":                "rs_1",
			"summary":           append([]any{}, summary...),
			"encrypted_content": "enc-final",
		}
	}

	for _, tc := range []struct {
		name      string
		store     bool
		encrypted *string
		summary   []string
		webSearch bool
		// want is the reasoning input item; nil means none is sent.
		want map[string]any
	}{
		{
			name:      "summary present store false",
			encrypted: new("enc-final"),
			summary:   []string{"first", "second"},
			want: fullItem(
				map[string]any{"type": "summary_text", "text": "first"},
				map[string]any{"type": "summary_text", "text": "second"},
			),
		},
		{
			name:      "empty summary store false",
			encrypted: new("enc-final"),
			summary:   []string{""},
			want:      fullItem(),
		},
		{
			name:      "store true",
			store:     true,
			encrypted: new("enc-final"),
			summary:   []string{"first"},
			want:      fullItem(map[string]any{"type": "summary_text", "text": "first"}),
		},
		{
			name:  "no blob store true",
			store: true,
			want:  reference,
		},
		{
			name: "no blob store false",
		},
		{
			name:      "store true before web search",
			store:     true,
			encrypted: new("enc-final"),
			webSearch: true,
			want:      fullItem(),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			next := fantasy.MessagePart(fantasy.ToolCallPart{ToolCallID: "call_1", ToolName: "add", Input: `{}`})
			wantNext := map[string]any{"type": "function_call", "call_id": "call_1"}
			if tc.webSearch {
				next = fantasy.ToolCallPart{ToolCallID: "ws_1", ToolName: "web_search", ProviderExecuted: true}
				wantNext = map[string]any{"id": "ws_1"}
			}
			prompt := fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "q"}}},
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesReasoningMetadata{
						ItemID:           "rs_1",
						EncryptedContent: tc.encrypted,
						Summary:          tc.summary,
						Finalized:        true,
					}}},
					next,
					fantasy.TextPart{Text: "a"},
				}},
			}
			if !tc.webSearch {
				prompt = append(prompt, fantasy.Message{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "call_1", Output: fantasy.ToolResultOutputContentText{Text: "2"}},
				}})
			}

			server := newMockServer()
			defer server.close()
			server.response = mockResponsesWebSearchResponse()
			_, err := newResponsesProvider(t, server.server.URL).Generate(context.Background(), fantasy.Call{
				Prompt:          prompt,
				ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesProviderOptions{Store: new(tc.store)}},
			})
			require.NoError(t, err)

			require.Len(t, server.calls, 1)
			input, ok := server.calls[0].body["input"].([]any)
			require.True(t, ok)
			rest := input[1:]
			if tc.want != nil {
				require.Equal(t, tc.want, rest[0])
				rest = rest[1:]
			}
			require.NotEmpty(t, rest)
			item, ok := rest[0].(map[string]any)
			require.True(t, ok)
			for key, value := range wantNext {
				require.Equal(t, value, item[key], "reasoning must stay immediately before %v", wantNext)
			}
		})
	}
}
