package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesStream_TextPhaseMetadata(t *testing.T) {
	t.Parallel()

	for _, phase := range []string{"commentary", "final_answer", ""} {
		t.Run(fmt.Sprintf("phase %q", phase), func(t *testing.T) {
			t.Parallel()

			phaseField := ""
			if phase != "" {
				phaseField = fmt.Sprintf(`,"phase":%q`, phase)
			}
			sms := newStreamingMockServer()
			defer sms.close()
			sms.chunks = []string{
				responsesSSEEvent("response.created", `{"type":"response.created","response":{"id":"resp_01","status":"in_progress","output":[]}}`),
				responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_01","type":"message","role":"assistant","status":"in_progress"`+phaseField+`,"content":[]}}`),
				responsesSSEEvent("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_01","delta":"Reading the file."}`),
				responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_01","type":"message","role":"assistant","status":"completed"`+phaseField+`,"content":[{"type":"output_text","text":"Reading the file.","annotations":[]}]}}`),
				responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
			}

			model := newResponsesProvider(t, sms.server.URL)
			stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)
			parts, err := collectStreamParts(stream)
			require.NoError(t, err)

			want := &ResponsesTextMetadata{ItemID: "msg_01", Phase: phase}
			var bounds []fantasy.StreamPartType
			for _, part := range parts {
				if part.Type != fantasy.StreamPartTypeTextStart && part.Type != fantasy.StreamPartTypeTextEnd {
					continue
				}
				bounds = append(bounds, part.Type)
				require.Equal(t, want, part.ProviderMetadata[Name], part.Type)
			}
			require.Equal(t, []fantasy.StreamPartType{fantasy.StreamPartTypeTextStart, fantasy.StreamPartTypeTextEnd}, bounds)
		})
	}
}

func TestResponsesGenerate_TextPhaseMetadata(t *testing.T) {
	t.Parallel()

	message := func(id, phase, text string) map[string]any {
		return map[string]any{
			"type":    "message",
			"id":      id,
			"role":    "assistant",
			"status":  "completed",
			"phase":   phase,
			"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
		}
	}
	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id":     "resp_01",
		"object": "response",
		"model":  "gpt-4.1",
		"status": "completed",
		"output": []any{
			message("msg_commentary", "commentary", "Reading the file."),
			message("msg_answer", "final_answer", "It is flaky."),
		},
		"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
	}

	model := newResponsesProvider(t, server.server.URL)
	resp, err := model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)

	var metadata []fantasy.ProviderOptionsData
	for _, content := range resp.Content {
		if text, ok := content.(fantasy.TextContent); ok {
			metadata = append(metadata, text.ProviderMetadata[Name])
		}
	}
	require.Equal(t, []fantasy.ProviderOptionsData{
		&ResponsesTextMetadata{ItemID: "msg_commentary", Phase: "commentary"},
		&ResponsesTextMetadata{ItemID: "msg_answer", Phase: "final_answer"},
	}, metadata)
}

func TestResponsesToPrompt_ReplaysTextPhase(t *testing.T) {
	t.Parallel()

	// Callers store the metadata as JSON and restore it through the registry.
	stored, err := json.Marshal(&ResponsesTextMetadata{ItemID: "msg_commentary", Phase: "commentary"})
	require.NoError(t, err)
	restored, err := fantasy.UnmarshalProviderOptions(map[string]json.RawMessage{Name: stored})
	require.NoError(t, err)

	prompt := fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Why is TestFoo flaky?"}},
		},
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "Reading the file.", ProviderOptions: restored},
				fantasy.TextPart{Text: "It is flaky.", ProviderOptions: fantasy.ProviderOptions{
					Name: &ResponsesTextMetadata{ItemID: "msg_answer", Phase: "final_answer"},
				}},
				fantasy.TextPart{Text: "Unlabeled."},
				fantasy.TextPart{Text: "No phase.", ProviderOptions: fantasy.ProviderOptions{
					Name: &ResponsesTextMetadata{ItemID: "msg_no_phase"},
				}},
			},
		},
	}

	for _, store := range []bool{true, false} {
		t.Run(fmt.Sprintf("store %t", store), func(t *testing.T) {
			t.Parallel()

			input, warnings, err := toResponsesPrompt(prompt, "system", store)
			require.NoError(t, err)
			require.Empty(t, warnings)
			require.Len(t, input, 5)

			var phases []any
			for _, item := range input[1:] {
				raw, err := json.Marshal(item)
				require.NoError(t, err)
				var fields map[string]any
				require.NoError(t, json.Unmarshal(raw, &fields))
				phases = append(phases, fields["phase"])
			}
			require.Equal(t, []any{"commentary", "final_answer", nil, nil}, phases)
		})
	}
}
