package openai

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestPrepareParams_ChatReasoningModelOverride(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		modelID       string
		opts          []Option
		wantReasoning bool
	}{
		{name: "unknown default", modelID: "totally-new-model"},
		{name: "gpt-5 default", modelID: "gpt-5", wantReasoning: true},
		{
			name: "force reasoning", modelID: "totally-new-model", wantReasoning: true,
			opts: []Option{WithReasoningModelFunc(func(modelID string) bool { return modelID == "totally-new-model" })},
		},
		{
			name: "force non-reasoning", modelID: "gpt-5",
			opts: []Option{WithReasoningModelFunc(func(modelID string) bool { return modelID != "gpt-5" })},
		},
		{
			name: "language model option", modelID: "totally-new-model", wantReasoning: true,
			opts: []Option{WithLanguageModelOptions(WithLanguageModelReasoningModelFunc(func(string) bool { return true }))},
		},
		{
			name: "provider option takes precedence", modelID: "gpt-5",
			opts: []Option{
				WithLanguageModelOptions(WithLanguageModelReasoningModelFunc(func(string) bool { return true })),
				WithReasoningModelFunc(func(string) bool { return false }),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := New(tt.opts...)
			require.NoError(t, err)
			model, err := provider.LanguageModel(context.Background(), tt.modelID)
			require.NoError(t, err)
			lm, ok := model.(languageModel)
			require.True(t, ok)

			params, warnings, err := lm.prepareParams(fantasy.Call{
				Prompt:          fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")},
				Temperature:     new(0.7),
				MaxOutputTokens: new(int64(512)),
				ProviderOptions: fantasy.ProviderOptions{
					Name: &ProviderOptions{LogProbs: new(true)},
				},
			})
			require.NoError(t, err)

			if tt.wantReasoning {
				require.False(t, params.Temperature.Valid())
				require.False(t, params.MaxTokens.Valid())
				require.True(t, params.MaxCompletionTokens.Valid())
				require.Equal(t, int64(512), params.MaxCompletionTokens.Value)
				require.False(t, params.Logprobs.Valid())
				var unsupported []string
				for _, warning := range warnings {
					require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warning.Type)
					unsupported = append(unsupported, warning.Setting)
				}
				require.ElementsMatch(t, []string{"temperature", "Logprobs"}, unsupported)
			} else {
				require.True(t, params.Temperature.Valid())
				require.Equal(t, 0.7, params.Temperature.Value)
				require.True(t, params.MaxTokens.Valid())
				require.Equal(t, int64(512), params.MaxTokens.Value)
				require.False(t, params.MaxCompletionTokens.Valid())
				require.True(t, params.Logprobs.Valid())
				require.True(t, params.Logprobs.Value)
				require.Empty(t, warnings)
			}
		})
	}
}
