package google

import (
	"encoding/base64"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

// Gemini function responses are JSON only, so media tool results are
// delivered as a text function response paired with a sibling inline
// data part in the same turn.

func mediaToolResultPrompt(output fantasy.ToolResultOutputContentMedia) fantasy.Prompt {
	return fantasy.Prompt{
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{ToolCallID: "call-1", ToolName: "screenshot", Input: "{}"},
			},
		},
		{
			Role: fantasy.MessageRoleTool,
			Content: []fantasy.MessagePart{
				fantasy.ToolResultPart{ToolCallID: "call-1", Output: output},
			},
		},
	}
}

func TestToGooglePrompt_MediaToolResult_ImageWithText(t *testing.T) {
	t.Parallel()

	raw := []byte{0, 1, 2, 3}
	prompt := mediaToolResultPrompt(fantasy.ToolResultOutputContentMedia{
		Data:      base64.StdEncoding.EncodeToString(raw),
		MediaType: "image/png",
		Text:      "Screenshot of the login page.",
	})

	_, contents, warnings := toGooglePrompt(prompt, false)

	require.Empty(t, warnings)
	require.Len(t, contents, 2)
	toolTurn := contents[1]
	require.Equal(t, genai.RoleUser, toolTurn.Role)
	require.Len(t, toolTurn.Parts, 2)

	response := toolTurn.Parts[0].FunctionResponse
	require.NotNil(t, response)
	require.Equal(t, "call-1", response.ID)
	require.Equal(t, "screenshot", response.Name)
	require.Equal(t, map[string]any{"result": "Screenshot of the login page."}, response.Response)

	inline := toolTurn.Parts[1].InlineData
	require.NotNil(t, inline)
	require.Equal(t, "image/png", inline.MIMEType)
	require.Equal(t, raw, inline.Data)
}

func TestToGooglePrompt_MediaToolResult_ImageWithoutText(t *testing.T) {
	t.Parallel()

	prompt := mediaToolResultPrompt(fantasy.ToolResultOutputContentMedia{
		Data:      base64.StdEncoding.EncodeToString([]byte{9, 9, 9}),
		MediaType: "image/jpeg",
	})

	_, contents, warnings := toGooglePrompt(prompt, true)

	require.Empty(t, warnings)
	require.Len(t, contents, 2)
	require.Len(t, contents[1].Parts, 2)

	response := contents[1].Parts[0].FunctionResponse
	require.NotNil(t, response)
	// Vertex rejects function response IDs.
	require.Empty(t, response.ID)
	require.Contains(t, response.Response["result"], "image/jpeg")
	require.NotNil(t, contents[1].Parts[1].InlineData)
}

func TestToGooglePrompt_MediaToolResult_InvalidBase64(t *testing.T) {
	t.Parallel()

	prompt := mediaToolResultPrompt(fantasy.ToolResultOutputContentMedia{
		Data:      "not base64!",
		MediaType: "image/png",
		Text:      "Screenshot text.",
	})

	_, contents, warnings := toGooglePrompt(prompt, false)

	require.Len(t, warnings, 1)
	require.Contains(t, warnings[0].Message, "not valid base64")
	require.Len(t, contents, 2)
	// The text function response still pairs with the tool call.
	require.Len(t, contents[1].Parts, 1)
	require.NotNil(t, contents[1].Parts[0].FunctionResponse)
	require.Equal(t, map[string]any{"result": "Screenshot text."}, contents[1].Parts[0].FunctionResponse.Response)
}
