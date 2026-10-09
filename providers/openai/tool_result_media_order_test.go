package openai_test

import (
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"
)

func TestToolMediaFollowsToolResultBatch(t *testing.T) {
	t.Parallel()

	for _, provider := range []struct {
		name     string
		toPrompt func(fantasy.Prompt, string, string) ([]openaisdk.ChatCompletionMessageParamUnion, []fantasy.CallWarning)
	}{
		{name: "OpenAI", toPrompt: openai.DefaultToPrompt},
		{name: "OpenAICompatible", toPrompt: openaicompat.ToPromptFunc},
	} {
		t.Run(provider.name, func(t *testing.T) {
			t.Parallel()
			for _, tc := range []struct {
				name             string
				separateMessages bool
				secondMedia      bool
				followup         bool
			}{
				{name: "SeparateMessagesMixed", separateMessages: true},
				{name: "SeparateMessagesTwoMedia", separateMessages: true, secondMedia: true},
				{name: "SameMessageMixed"},
				{name: "SameMessageTwoMedia", secondMedia: true},
				{name: "BeforeFollowingMessages", separateMessages: true, secondMedia: true, followup: true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					media := fantasy.ToolResultOutputContentMedia{Data: "AAEC", MediaType: "image/png", Text: "first result"}
					results := []fantasy.MessagePart{
						fantasy.ToolResultPart{ToolCallID: "call-a", Output: media},
						fantasy.ToolResultPart{ToolCallID: "call-b", Output: fantasy.ToolResultOutputContentText{Text: "second result"}},
					}
					imageURLs := []string{"data:image/png;base64,AAEC"}
					if tc.secondMedia {
						media.Data, media.Text = "AwQF", "second result"
						results[1] = fantasy.ToolResultPart{ToolCallID: "call-b", Output: media}
						imageURLs = append(imageURLs, "data:image/png;base64,AwQF")
					}
					prompt := fantasy.Prompt{{
						Role: fantasy.MessageRoleAssistant,
						Content: []fantasy.MessagePart{
							fantasy.ToolCallPart{ToolCallID: "call-a", ToolName: "first", Input: "{}"},
							fantasy.ToolCallPart{ToolCallID: "call-b", ToolName: "second", Input: "{}"},
						},
					}}
					if tc.separateMessages {
						for _, result := range results {
							prompt = append(prompt, fantasy.Message{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{result}})
						}
					} else {
						prompt = append(prompt, fantasy.Message{Role: fantasy.MessageRoleTool, Content: results})
					}
					if tc.followup {
						prompt = append(prompt,
							fantasy.Message{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "done"}}},
							fantasy.Message{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "thanks"}}},
						)
					}

					messages, warnings := provider.toPrompt(prompt, "", "")
					require.Empty(t, warnings)
					wantLen := 3 + len(imageURLs)
					if tc.followup {
						wantLen += 2
					}
					require.Len(t, messages, wantLen)
					require.NotNil(t, messages[0].OfAssistant)
					require.Len(t, messages[0].OfAssistant.ToolCalls, 2)
					for i, want := range []struct{ id, text string }{{"call-a", "first result"}, {"call-b", "second result"}} {
						tool := messages[i+1].OfTool
						require.NotNil(t, tool, "all tool results must precede synthetic user media")
						require.Equal(t, want.id, tool.ToolCallID)
						require.Equal(t, want.text, tool.Content.OfString.Value)
					}
					for i, url := range imageURLs {
						user := messages[3+i].OfUser
						require.NotNil(t, user)
						require.Len(t, user.Content.OfArrayOfContentParts, 1)
						image := user.Content.OfArrayOfContentParts[0].OfImageURL
						require.NotNil(t, image)
						require.Equal(t, url, image.ImageURL.URL)
					}
					if tc.followup {
						require.NotNil(t, messages[wantLen-2].OfAssistant)
						require.Equal(t, "done", messages[wantLen-2].OfAssistant.Content.OfString.Value)
						require.NotNil(t, messages[wantLen-1].OfUser)
						require.Equal(t, "thanks", messages[wantLen-1].OfUser.Content.OfString.Value)
					}
				})
			}
		})
	}
}
