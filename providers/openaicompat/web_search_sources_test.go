package openaicompat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
)

func TestWithoutWebSearchSources(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name        string
		opts        []Option
		wantInclude bool
	}{
		{name: "Default", wantInclude: true},
		{name: "OptedOut", opts: []Option{WithoutWebSearchSources()}, wantInclude: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var (
				mu      sync.Mutex
				include []string
			)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Include []string `json:"include"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				mu.Lock()
				include = body.Include
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"resp_01","object":"response","status":"completed","model":"gpt-4.1","output":[{"type":"message","id":"msg_01","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}]}`))
			}))
			defer server.Close()

			provider, err := New(append([]Option{
				WithBaseURL(server.URL),
				WithAPIKey("test-api-key"),
				WithUseResponsesAPI(),
			}, tc.opts...)...)
			require.NoError(t, err)
			model, err := provider.LanguageModel(context.Background(), "gpt-4.1")
			require.NoError(t, err)

			_, err = model.Generate(context.Background(), fantasy.Call{
				Prompt: fantasy.Prompt{{
					Role:    fantasy.MessageRoleUser,
					Content: []fantasy.MessagePart{fantasy.TextPart{Text: "search"}},
				}},
				Tools: []fantasy.Tool{openai.WebSearchTool(nil)},
			})
			require.NoError(t, err)

			mu.Lock()
			defer mu.Unlock()
			if tc.wantInclude {
				require.Contains(t, include, string(openai.IncludeWebSearchCallActionSources))
			} else {
				require.NotContains(t, include, string(openai.IncludeWebSearchCallActionSources))
			}
		})
	}
}
