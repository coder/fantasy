package google

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestToolInputSchemaReachesRequest(t *testing.T) {
	t.Parallel()

	// Each part of this schema is something the OpenAPI-style
	// "parameters" field cannot carry: an untyped object root, local
	// references with sibling keywords, a recursive definition, a
	// numeric enum, a null alternative behind a reference, and
	// additionalProperties.
	const schemaJSON = `{
		"properties": {
			"filter": {"$ref": "#/$defs/Filter", "required": ["term"]},
			"limit": {"type": "integer", "enum": [10, 20]},
			"cursor": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/Null"}]}
		},
		"required": ["filter"],
		"additionalProperties": false,
		"$defs": {
			"Filter": {
				"type": "object",
				"properties": {
					"term": {"type": "string"},
					"and": {"type": "array", "items": {"$ref": "#/$defs/Filter"}}
				}
			},
			"Null": {"type": "null"}
		}
	}`

	for _, tc := range []struct {
		name    string
		options []Option
	}{
		{name: "gemini", options: []Option{WithGeminiAPIKey("test-key")}},
		{name: "vertex", options: []Option{WithVertex("test-project", "us-central1"), WithSkipAuth(true)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var inputSchema map[string]any
			require.NoError(t, json.Unmarshal([]byte(schemaJSON), &inputSchema))

			bodies := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				bodies <- body
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
			}))
			defer server.Close()

			provider, err := New(append(tc.options, WithBaseURL(server.URL))...)
			require.NoError(t, err)
			model, err := provider.LanguageModel(t.Context(), "gemini-2.5-flash")
			require.NoError(t, err)

			_, err = model.Generate(t.Context(), fantasy.Call{
				Prompt: fantasy.Prompt{{
					Role:    fantasy.MessageRoleUser,
					Content: []fantasy.MessagePart{fantasy.TextPart{Text: "search"}},
				}},
				Tools: []fantasy.Tool{
					fantasy.FunctionTool{Name: "search", InputSchema: inputSchema},
					fantasy.FunctionTool{Name: "ping"},
				},
			})
			require.NoError(t, err)

			var request struct {
				Tools []struct {
					FunctionDeclarations []map[string]json.RawMessage `json:"functionDeclarations"`
				} `json:"tools"`
			}
			require.NoError(t, json.Unmarshal(<-bodies, &request))
			require.Len(t, request.Tools, 1)
			declarations := request.Tools[0].FunctionDeclarations
			require.Len(t, declarations, 2)

			require.NotContains(t, declarations[0], "parameters")
			require.JSONEq(t, schemaJSON, string(declarations[0]["parametersJsonSchema"]))

			// A tool without a schema declares no parameters rather than
			// a null schema.
			require.NotContains(t, declarations[1], "parameters")
			require.NotContains(t, declarations[1], "parametersJsonSchema")
		})
	}
}
