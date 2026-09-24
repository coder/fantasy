package openai

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// webSearchCallActionCases covers the web_search_call action shapes
// OpenAI returns: current responses with queries and consulted sources,
// older responses with the deprecated single query, and searches that
// report no query at all.
var webSearchCallActionCases = []struct {
	name      string
	action    map[string]any
	want      *WebSearchAction
	wantInput string
}{
	{
		name: "QueriesAndSources",
		action: map[string]any{
			"type":    "search",
			"queries": []any{"coder agents web search", "openai responses sources"},
			"sources": []any{
				map[string]any{"type": "url", "url": "https://coder.com/docs/ai-coder"},
				map[string]any{"type": "url", "url": "https://developers.openai.com/api/docs/guides/tools-web-search"},
			},
		},
		want: &WebSearchAction{
			Type:    "search",
			Queries: []string{"coder agents web search", "openai responses sources"},
			Sources: []WebSearchSource{
				{Type: "url", URL: "https://coder.com/docs/ai-coder"},
				{Type: "url", URL: "https://developers.openai.com/api/docs/guides/tools-web-search"},
			},
		},
		wantInput: `{"queries":["coder agents web search","openai responses sources"]}`,
	},
	{
		name: "DeprecatedQuery",
		action: map[string]any{
			"type":  "search",
			"query": "latest AI news",
		},
		want: &WebSearchAction{
			Type:  "search",
			Query: "latest AI news",
		},
		wantInput: `{"queries":["latest AI news"]}`,
	},
	{
		name: "NoQueries",
		action: map[string]any{
			"type": "search",
			"sources": []any{
				map[string]any{"type": "url", "url": "https://example.com/weather"},
			},
		},
		want: &WebSearchAction{
			Type: "search",
			Sources: []WebSearchSource{
				{Type: "url", URL: "https://example.com/weather"},
			},
		},
		wantInput: `{"queries":[]}`,
	},
}

func requireWebSearchCallMetadata(t *testing.T, metadata fantasy.ProviderMetadata, want *WebSearchAction) {
	t.Helper()

	wsMeta, ok := metadata[Name].(*WebSearchCallMetadata)
	require.True(t, ok, "metadata should be *WebSearchCallMetadata, got %T", metadata[Name])
	require.Equal(t, "ws_01", wsMeta.ItemID)
	require.Equal(t, want, wsMeta.Action)

	// Callers persist the metadata as JSON and read it back on later
	// turns, so the action fields must survive the round trip.
	encoded, err := json.Marshal(metadata)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encoded, &raw))
	decoded, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	roundTripped, ok := decoded[Name].(*WebSearchCallMetadata)
	require.True(t, ok, "decoded metadata should be *WebSearchCallMetadata, got %T", decoded[Name])
	require.Equal(t, wsMeta, roundTripped)
}

func requireWebSearchSourcesInclude(t *testing.T, body map[string]any) {
	t.Helper()

	include, ok := body["include"].([]any)
	require.True(t, ok, "request body should have an include array, got %#v", body["include"])
	count := 0
	for _, value := range include {
		if value == string(IncludeWebSearchCallActionSources) {
			count++
		}
	}
	require.Equal(t, 1, count, "include should request web_search_call.action.sources exactly once: %v", include)
}

func TestResponsesGenerate_WebSearchCallAction(t *testing.T) {
	t.Parallel()

	for _, tc := range webSearchCallActionCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := newMockServer()
			defer server.close()
			server.response = map[string]any{
				"id":     "resp_01",
				"object": "response",
				"model":  "gpt-4.1",
				"status": "completed",
				"output": []any{
					map[string]any{
						"type":   "web_search_call",
						"id":     "ws_01",
						"status": "completed",
						"action": tc.action,
					},
					map[string]any{
						"type":   "message",
						"id":     "msg_01",
						"role":   "assistant",
						"status": "completed",
						"content": []any{
							map[string]any{
								"type": "output_text",
								"text": "Answer.",
								"annotations": []any{
									map[string]any{
										"type":        "url_citation",
										"url":         "https://example.com/cited",
										"title":       "Cited",
										"start_index": 0,
										"end_index":   7,
									},
								},
							},
						},
					},
				},
				"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
			}

			model := newResponsesProvider(t, server.server.URL)
			resp, err := model.Generate(context.Background(), fantasy.Call{
				Prompt: testPrompt,
				Tools:  []fantasy.Tool{WebSearchTool(nil)},
			})
			require.NoError(t, err)
			require.Len(t, server.calls, 1)
			requireWebSearchSourcesInclude(t, server.calls[0].body)

			var (
				toolCalls   []fantasy.ToolCallContent
				toolResults []fantasy.ToolResultContent
				sources     []fantasy.SourceContent
			)
			for _, c := range resp.Content {
				switch v := c.(type) {
				case fantasy.ToolCallContent:
					toolCalls = append(toolCalls, v)
				case fantasy.ToolResultContent:
					toolResults = append(toolResults, v)
				case fantasy.SourceContent:
					sources = append(sources, v)
				}
			}

			require.Len(t, toolCalls, 1)
			require.True(t, toolCalls[0].ProviderExecuted)
			require.JSONEq(t, tc.wantInput, toolCalls[0].Input)
			require.Len(t, toolResults, 1)
			require.True(t, toolResults[0].ProviderExecuted)
			require.Equal(t, "web_search", toolResults[0].ToolName)
			requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, tc.want)

			// Consulted sources must not become citation sources.
			require.Len(t, sources, 1)
			require.Equal(t, "https://example.com/cited", sources[0].URL)
		})
	}
}

func TestResponsesStream_WebSearchCallAction(t *testing.T) {
	t.Parallel()

	for _, tc := range webSearchCallActionCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			item, err := json.Marshal(map[string]any{
				"type":   "web_search_call",
				"id":     "ws_01",
				"status": "completed",
				"action": tc.action,
			})
			require.NoError(t, err)

			sms := newStreamingMockServer()
			defer sms.close()
			sms.chunks = []string{
				responsesSSEEvent("response.output_item.added",
					`{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"in_progress"}}`),
				responsesSSEEvent("response.output_item.done",
					`{"type":"response.output_item.done","output_index":0,"item":`+string(item)+`}`),
				responsesSSEEvent("response.output_item.added",
					`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"in_progress","content":[]}}`),
				responsesSSEEvent("response.output_text.delta",
					`{"type":"response.output_text.delta","item_id":"msg_01","output_index":1,"content_index":0,"delta":"Answer."}`),
				responsesSSEEvent("response.output_text.annotation.added",
					`{"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://example.com/cited","title":"Cited","start_index":0,"end_index":7},"annotation_index":0,"content_index":0,"item_id":"msg_01","output_index":1,"sequence_number":5}`),
				responsesSSEEvent("response.output_item.done",
					`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Answer.","annotations":[]}]}}`),
				responsesSSEEvent("response.completed",
					`{"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
			}

			model := newResponsesProvider(t, sms.server.URL)
			stream, err := model.Stream(context.Background(), fantasy.Call{
				Prompt: testPrompt,
				Tools:  []fantasy.Tool{WebSearchTool(nil)},
			})
			require.NoError(t, err)

			var toolCalls, toolResults, sources []fantasy.StreamPart
			for part := range stream {
				require.NotEqual(t, fantasy.StreamPartTypeError, part.Type, "unexpected stream error: %v", part.Error)
				switch part.Type {
				case fantasy.StreamPartTypeToolCall:
					toolCalls = append(toolCalls, part)
				case fantasy.StreamPartTypeToolResult:
					toolResults = append(toolResults, part)
				case fantasy.StreamPartTypeSource:
					sources = append(sources, part)
				}
			}
			require.Len(t, sms.calls, 1)
			requireWebSearchSourcesInclude(t, sms.calls[0].body)

			require.Len(t, toolCalls, 1)
			require.True(t, toolCalls[0].ProviderExecuted)
			require.JSONEq(t, tc.wantInput, toolCalls[0].ToolCallInput)
			require.Len(t, toolResults, 1)
			require.True(t, toolResults[0].ProviderExecuted)
			require.Equal(t, "web_search", toolResults[0].ToolCallName)
			requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, tc.want)

			// Consulted sources must not become citation sources.
			require.Len(t, sources, 1)
			require.Equal(t, "https://example.com/cited", sources[0].URL)
		})
	}
}

// TestResponsesStream_WebSearchCallSourcesFromCompletedResponse covers the
// recorded OpenAI behavior where response.output_item.done omits
// action.sources and only response.completed lists them.
func TestResponsesStream_WebSearchCallSourcesFromCompletedResponse(t *testing.T) {
	t.Parallel()

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = []string{
		responsesSSEEvent("response.output_item.added",
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"in_progress","action":{"type":"search"}}}`),
		responsesSSEEvent("response.output_item.done",
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"],"query":"tokyo population"}}}`),
		responsesSSEEvent("response.output_item.added",
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"in_progress","content":[]}}`),
		responsesSSEEvent("response.output_text.delta",
			`{"type":"response.output_text.delta","item_id":"msg_01","output_index":1,"content_index":0,"delta":"About 14 million."}`),
		responsesSSEEvent("response.output_item.done",
			`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"About 14 million.","annotations":[]}]}}`),
		responsesSSEEvent("response.completed",
			`{"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[`+
				`{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"],"query":"tokyo population","sources":[{"type":"url","url":"https://www.metro.tokyo.lg.jp/"},{"type":"url","url":"https://example.com/tokyo"}]}},`+
				`{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"About 14 million.","annotations":[]}]}`+
				`],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
	}

	model := newResponsesProvider(t, sms.server.URL)
	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools:  []fantasy.Tool{WebSearchTool(nil)},
	})
	require.NoError(t, err)

	var types []fantasy.StreamPartType
	var toolResults []fantasy.StreamPart
	for part := range stream {
		require.NotEqual(t, fantasy.StreamPartTypeError, part.Type, "unexpected stream error: %v", part.Error)
		types = append(types, part.Type)
		if part.Type == fantasy.StreamPartTypeToolResult {
			toolResults = append(toolResults, part)
		}
	}

	// The call keeps its position ahead of the answer text; the
	// result follows once the terminal event arrives.
	require.Less(t, slices.Index(types, fantasy.StreamPartTypeToolCall), slices.Index(types, fantasy.StreamPartTypeTextDelta))
	require.Less(t, slices.Index(types, fantasy.StreamPartTypeTextDelta), slices.Index(types, fantasy.StreamPartTypeToolResult))
	require.Less(t, slices.Index(types, fantasy.StreamPartTypeToolResult), slices.Index(types, fantasy.StreamPartTypeFinish))

	require.Len(t, toolResults, 1)
	requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, &WebSearchAction{
		Type:    "search",
		Queries: []string{"tokyo population"},
		Query:   "tokyo population",
		Sources: []WebSearchSource{
			{Type: "url", URL: "https://www.metro.tokyo.lg.jp/"},
			{Type: "url", URL: "https://example.com/tokyo"},
		},
	})
}

func TestResponsesStream_WebSearchCallResultWithoutTerminalEvent(t *testing.T) {
	t.Parallel()

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = []string{
		responsesSSEEvent("response.output_item.done",
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"]}}}`),
	}

	model := newResponsesProvider(t, sms.server.URL)
	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools:  []fantasy.Tool{WebSearchTool(nil)},
	})
	require.NoError(t, err)

	var toolResults, errs []fantasy.StreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.StreamPartTypeToolResult:
			require.Empty(t, errs, "the paired result must precede the stream error")
			toolResults = append(toolResults, part)
		case fantasy.StreamPartTypeError:
			errs = append(errs, part)
		}
	}

	require.Len(t, errs, 1)
	require.Len(t, toolResults, 1)
	requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, &WebSearchAction{
		Type:    "search",
		Queries: []string{"tokyo population"},
	})
}

func TestResponsesStream_WebSearchCallSourcesFromFailedResponse(t *testing.T) {
	t.Parallel()

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = []string{
		responsesSSEEvent("response.output_item.done",
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"]}}}`),
		responsesSSEEvent("response.failed",
			`{"type":"response.failed","response":{"id":"resp_01","status":"failed","error":{"code":"server_error","message":"boom"},"output":[`+
				`{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"],"sources":[{"type":"url","url":"https://www.metro.tokyo.lg.jp/"}]}}`+
				`]}}`),
	}

	model := newResponsesProvider(t, sms.server.URL)
	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools:  []fantasy.Tool{WebSearchTool(nil)},
	})
	require.NoError(t, err)

	var toolResults, errs []fantasy.StreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.StreamPartTypeToolResult:
			require.Empty(t, errs, "the paired result must precede the stream error")
			toolResults = append(toolResults, part)
		case fantasy.StreamPartTypeError:
			errs = append(errs, part)
		}
	}

	require.Len(t, errs, 1)
	require.Len(t, toolResults, 1)
	requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, &WebSearchAction{
		Type:    "search",
		Queries: []string{"tokyo population"},
		Sources: []WebSearchSource{{Type: "url", URL: "https://www.metro.tokyo.lg.jp/"}},
	})
}

func TestResponsesWebSearchSourcesInclude(t *testing.T) {
	t.Parallel()

	t.Run("NotRequestedWithoutWebSearchTool", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)
		_, err := model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		include, _ := server.calls[0].body["include"].([]any)
		require.NotContains(t, include, string(IncludeWebSearchCallActionSources))
	})

	t.Run("KeepsCallerIncludesWithoutDuplicates", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)
		_, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools:  []fantasy.Tool{WebSearchTool(nil)},
			ProviderOptions: fantasy.ProviderOptions{
				Name: &ResponsesProviderOptions{
					Include: []IncludeType{
						IncludeReasoningEncryptedContent,
						IncludeWebSearchCallActionSources,
					},
				},
			},
		})
		require.NoError(t, err)
		require.Len(t, server.calls, 1)
		requireWebSearchSourcesInclude(t, server.calls[0].body)
		require.Contains(t, server.calls[0].body["include"], string(IncludeReasoningEncryptedContent))
	})
}
