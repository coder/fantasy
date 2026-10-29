package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestComputerUseToolVersionForModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model string
		want  ComputerUseToolVersion
	}{
		{"claude-opus-5-5", ComputerUseToolset20260801},
		{"claude-sonnet-5-5", ComputerUseToolset20260801},
		{"claude-haiku-5-5", ComputerUseToolset20260801},
		{"claude-fable-5-1", ComputerUseToolset20260801},
		{"claude-mythos-5", ComputerUseToolset20260801},
		{"claude-mythos-preview", ComputerUseToolset20260801},
		{"claude-opus-4-8", ComputerUseToolset20260801},
		{"anthropic/claude-opus-5-5", ComputerUseToolset20260801},
		{"my-custom-alias", ComputerUseToolset20260801},
		{"claude-opus-4-7", ComputerUse20251124},
		{"claude-opus-4-7@20260101", ComputerUse20251124},
		{"claude-opus-4-6", ComputerUse20251124},
		{"claude-sonnet-4-6", ComputerUse20251124},
		{"us.anthropic.claude-sonnet-4-6", ComputerUse20251124},
		{"claude-opus-4-5-20251101", ComputerUse20251124},
		{"claude-sonnet-4-5-20250929", ComputerUse20250124},
		{"claude-haiku-4-5-20251001", ComputerUse20250124},
		{"claude-opus-4-1", ComputerUse20250124},
		{"claude-sonnet-4-20250514", ComputerUse20250124},
		{"claude-3-7-sonnet-20250219", ComputerUse20250124},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tt.want, ComputerUseToolVersionForModel(tt.model))
		})
	}
}

func newComputerToolset(t *testing.T, opts ComputerUseToolOptions) fantasy.ProviderDefinedTool {
	t.Helper()
	opts.ToolVersion = ComputerUseToolset20260801
	return jsonRoundTripTool(t, NewComputerUseTool(opts, noopComputerRun))
}

func TestComputerToolsetToolJSON(t *testing.T) {
	t.Parallel()

	zoomOff := false
	displayNumber := int64(1)
	tests := []struct {
		name string
		opts ComputerUseToolOptions
		want string
	}{
		{
			name: "defaults omit display fields and configs",
			opts: ComputerUseToolOptions{DisplayWidthPx: 1920, DisplayHeightPx: 1080, DisplayNumber: &displayNumber},
			want: `{"type":"computer_toolset_20260801"}`,
		},
		{
			name: "zoom disabled and cache control",
			opts: ComputerUseToolOptions{EnableZoom: &zoomOff, CacheControl: &CacheControl{Type: "ephemeral"}},
			want: `{"type":"computer_toolset_20260801","configs":{"zoom":{"enabled":false}},"cache_control":{"type":"ephemeral"}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			lm := languageModel{options: options{}}
			rawTools, _, warnings, betaFlags := lm.toTools([]fantasy.Tool{newComputerToolset(t, tt.opts)}, nil, false)
			require.Empty(t, warnings)
			require.Empty(t, betaFlags)
			require.Len(t, rawTools, 1)
			require.JSONEq(t, tt.want, string(rawTools[0]))
		})
	}
}

func computerToolsetMessageResponse(content []any) map[string]any {
	resp := mockAnthropicGenerateResponse()
	resp["content"] = content
	resp["stop_reason"] = "tool_use"
	return resp
}

type capturedRequest struct {
	header   http.Header
	rawQuery string
	body     map[string]any
}

func newCapturingServer(t *testing.T, response map[string]any) (*httptest.Server, <-chan capturedRequest) {
	t.Helper()
	requests := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		requests <- capturedRequest{header: r.Header.Clone(), rawQuery: r.URL.RawQuery, body: body}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	return server, requests
}

func generateWithServer(t *testing.T, serverURL string, call fantasy.Call) *fantasy.Response {
	t.Helper()
	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(serverURL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "claude-opus-5-5")
	require.NoError(t, err)
	resp, err := model.Generate(context.Background(), call)
	require.NoError(t, err)
	return resp
}

func TestGenerate_ComputerToolset(t *testing.T) {
	t.Parallel()

	server, requests := newCapturingServer(t, computerToolsetMessageResponse([]any{
		map[string]any{
			"type":         "tool_use",
			"id":           "toolu_click",
			"name":         "left_click",
			"toolset_name": "computer",
			"input":        map[string]any{"coordinate": []int{640, 60}},
		},
		map[string]any{
			"type":         "tool_use",
			"id":           "toolu_key",
			"name":         "key",
			"toolset_name": "computer",
			"input":        map[string]any{"text": "Tab", "repeat": 4},
		},
		map[string]any{
			"type":  "tool_use",
			"id":    "toolu_weather",
			"name":  "weather",
			"input": map[string]any{"location": "Paris"},
		},
	}))

	resp := generateWithServer(t, server.URL, fantasy.Call{
		Prompt: testPrompt(),
		Tools: []fantasy.Tool{
			newComputerToolset(t, ComputerUseToolOptions{DisplayWidthPx: 1920, DisplayHeightPx: 1080}),
			fantasy.FunctionTool{Name: "weather", InputSchema: map[string]any{"properties": map[string]any{}}},
		},
	})

	req := <-requests
	require.Empty(t, req.header.Values("Anthropic-Beta"))
	require.NotContains(t, req.rawQuery, "beta")
	tools, err := json.Marshal(req.body["tools"].([]any)[0])
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"computer_toolset_20260801"}`, string(tools))

	toolCalls := resp.Content.ToolCalls()
	require.Len(t, toolCalls, 3)
	require.Equal(t, "toolu_click", toolCalls[0].ToolCallID)
	require.Equal(t, "computer", toolCalls[0].ToolName)
	require.JSONEq(t, `{"action":"left_click","coordinate":[640,60]}`, toolCalls[0].Input)
	require.Equal(t, "computer", toolCalls[1].ToolName)
	parsed, err := ParseComputerUseInput(toolCalls[1].Input)
	require.NoError(t, err)
	require.Equal(t, ComputerUseInput{Action: ActionKey, Text: "Tab", Repeat: 4}, parsed)
	require.Equal(t, "weather", toolCalls[2].ToolName)
	require.JSONEq(t, `{"location":"Paris"}`, toolCalls[2].Input)
}

func TestStream_ComputerToolsetBatch(t *testing.T) {
	t.Parallel()

	chunks := []string{
		anthropicSSEEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`),
		anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_click","name":"left_click","toolset_name":"computer","input":{}}}`),
		anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"coordinate\":"}}`),
		anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"[640,60]}"}}`),
		anthropicSSEEvent("content_block_stop", `{"type":"content_block_stop","index":0}`),
		anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_shot","name":"screenshot","toolset_name":"computer","input":{}}}`),
		anthropicSSEEvent("content_block_stop", `{"type":"content_block_stop","index":1}`),
		anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_weather","name":"weather","input":{}}}`),
		anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"Paris\"}"}}`),
		anthropicSSEEvent("content_block_stop", `{"type":"content_block_stop","index":2}`),
		anthropicSSEEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":20}}`),
		anthropicSSEEvent("message_stop", `{"type":"message_stop"}`),
	}

	parts := streamAnthropicParts(t, chunks,
		newComputerToolset(t, ComputerUseToolOptions{}),
		fantasy.FunctionTool{Name: "weather", InputSchema: map[string]any{"properties": map[string]any{}}},
	)

	starts := streamPartsByType(parts, fantasy.StreamPartTypeToolInputStart)
	require.Len(t, starts, 3)
	require.Equal(t, []string{"computer", "computer", "weather"},
		[]string{starts[0].ToolCallName, starts[1].ToolCallName, starts[2].ToolCallName})

	deltas := map[string]string{}
	for _, part := range streamPartsByType(parts, fantasy.StreamPartTypeToolInputDelta) {
		deltas[part.ID] += part.ToolCallInput
	}
	require.Len(t, streamPartsByType(parts, fantasy.StreamPartTypeToolInputEnd), 3)

	calls := streamPartsByType(parts, fantasy.StreamPartTypeToolCall)
	require.Len(t, calls, 3)
	want := []struct{ id, name, input string }{
		{"toolu_click", "computer", `{"action":"left_click","coordinate":[640,60]}`},
		{"toolu_shot", "computer", `{"action":"screenshot"}`},
		{"toolu_weather", "weather", `{"location":"Paris"}`},
	}
	for i, w := range want {
		require.Equal(t, w.id, calls[i].ID)
		require.Equal(t, w.name, calls[i].ToolCallName)
		require.JSONEq(t, w.input, calls[i].ToolCallInput)
		// Streamed deltas must add up to the reported input.
		require.Equal(t, calls[i].ToolCallInput, deltas[w.id])
	}
}

// computerBatchPrompt is a turn where the model ran a two-action
// computer batch next to a regular tool call, followed by the results.
func computerBatchPrompt(clickInput string) fantasy.Prompt {
	return fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Search for cats"}}},
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ToolCallPart{ToolCallID: "toolu_click", ToolName: "computer", Input: clickInput},
			fantasy.ToolCallPart{ToolCallID: "toolu_shot", ToolName: "computer", Input: `{"action":"screenshot"}`},
			fantasy.ToolCallPart{ToolCallID: "toolu_weather", ToolName: "weather", Input: `{"location":"Paris"}`},
		}},
		{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
			NewComputerUseTextResult("toolu_click", "OK"),
			NewComputerUseScreenshotResultWithMediaType("toolu_shot", "aW1n", "image/png"),
			fantasy.ToolResultPart{ToolCallID: "toolu_weather", Output: fantasy.ToolResultOutputContentText{Text: "sunny"}},
		}},
	}
}

func replayedBlocks(t *testing.T, body map[string]any) (assistant, user string) {
	t.Helper()
	messages := body["messages"].([]any)
	require.Len(t, messages, 3)
	a, err := json.Marshal(messages[1].(map[string]any)["content"])
	require.NoError(t, err)
	u, err := json.Marshal(messages[2].(map[string]any)["content"])
	require.NoError(t, err)
	return string(a), string(u)
}

func TestReplay_ComputerToolset(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		version       ComputerUseToolVersion
		clickInput    string
		wantAssistant string
		wantUser      string
		wantWarning   bool
	}{
		{
			name:       "toolset members carry toolset_name",
			version:    ComputerUseToolset20260801,
			clickInput: `{"action":"left_click","coordinate":[640,60]}`,
			wantAssistant: `[
				{"type":"tool_use","id":"toolu_click","name":"left_click","toolset_name":"computer","input":{"coordinate":[640,60]}},
				{"type":"tool_use","id":"toolu_shot","name":"screenshot","toolset_name":"computer","input":{}},
				{"type":"tool_use","id":"toolu_weather","name":"weather","input":{"location":"Paris"}}
			]`,
			wantUser: `[
				{"type":"tool_result","tool_use_id":"toolu_click","toolset_name":"computer","content":[{"type":"text","text":"OK"}]},
				{"type":"tool_result","tool_use_id":"toolu_shot","toolset_name":"computer","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1n"}}]},
				{"type":"tool_result","tool_use_id":"toolu_weather","content":[{"type":"text","text":"sunny"}]}
			]`,
		},
		{
			name:        "toolset call without action falls back to screenshot",
			version:     ComputerUseToolset20260801,
			clickInput:  `{"coordinate":[640,60]}`,
			wantWarning: true,
			wantAssistant: `[
				{"type":"tool_use","id":"toolu_click","name":"screenshot","toolset_name":"computer","input":{}},
				{"type":"tool_use","id":"toolu_shot","name":"screenshot","toolset_name":"computer","input":{}},
				{"type":"tool_use","id":"toolu_weather","name":"weather","input":{"location":"Paris"}}
			]`,
			wantUser: `[
				{"type":"tool_result","tool_use_id":"toolu_click","toolset_name":"computer","content":[{"type":"text","text":"OK"}]},
				{"type":"tool_result","tool_use_id":"toolu_shot","toolset_name":"computer","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1n"}}]},
				{"type":"tool_result","tool_use_id":"toolu_weather","content":[{"type":"text","text":"sunny"}]}
			]`,
		},
		{
			name:       "computer_20251124 replay is unchanged",
			version:    ComputerUse20251124,
			clickInput: `{"action":"left_click","coordinate":[640,60]}`,
			wantAssistant: `[
				{"type":"tool_use","id":"toolu_click","name":"computer","input":{"action":"left_click","coordinate":[640,60]}},
				{"type":"tool_use","id":"toolu_shot","name":"computer","input":{"action":"screenshot"}},
				{"type":"tool_use","id":"toolu_weather","name":"weather","input":{"location":"Paris"}}
			]`,
			wantUser: `[
				{"type":"tool_result","tool_use_id":"toolu_click","content":[{"type":"text","text":"OK"}]},
				{"type":"tool_result","tool_use_id":"toolu_shot","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1n"}}]},
				{"type":"tool_result","tool_use_id":"toolu_weather","content":[{"type":"text","text":"sunny"}]}
			]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server, requests := newCapturingServer(t, mockAnthropicGenerateResponse())
			computer := jsonRoundTripTool(t, NewComputerUseTool(ComputerUseToolOptions{
				DisplayWidthPx: 1920, DisplayHeightPx: 1080, ToolVersion: tt.version,
			}, noopComputerRun))
			resp := generateWithServer(t, server.URL, fantasy.Call{
				Prompt: computerBatchPrompt(tt.clickInput),
				Tools: []fantasy.Tool{
					computer,
					fantasy.FunctionTool{Name: "weather", InputSchema: map[string]any{"properties": map[string]any{}}},
				},
			})

			assistant, user := replayedBlocks(t, (<-requests).body)
			require.JSONEq(t, tt.wantAssistant, assistant)
			require.JSONEq(t, tt.wantUser, user)

			if !tt.wantWarning {
				require.Empty(t, resp.Warnings)
				return
			}
			require.Len(t, resp.Warnings, 1)
			require.Contains(t, resp.Warnings[0].Message, `"toolu_click" has no string action`)
		})
	}
}
