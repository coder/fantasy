package google

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestToGoogleTools_ConvertsFunctionInputSchema(t *testing.T) {
	t.Parallel()

	tools := []fantasy.Tool{
		fantasy.FunctionTool{
			Name: "schema_test",
			InputSchema: map[string]any{
				"type": "object",
				"$defs": map[string]any{
					"node": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"kind": map[string]any{
								"type": "string",
								"enum": []any{"leaf", "branch"},
							},
							"child": map[string]any{"$ref": "#/$defs/node"},
						},
						"required": []any{"kind"},
					},
				},
				"properties": map[string]any{
					"node": map[string]any{"$ref": "#/$defs/node"},
					"choice": map[string]any{
						"anyOf": []any{
							map[string]any{"type": "string", "enum": []any{"a", "b"}},
							map[string]any{
								"type": "object",
								"properties": map[string]any{
									"count": map[string]any{"type": "integer"},
								},
								"required": []any{"count"},
							},
						},
					},
				},
				"required": []any{"node", "choice"},
			},
		},
	}

	declarations, _, warnings := toGoogleTools(tools, nil)
	require.Empty(t, warnings)
	require.Len(t, declarations, 1)

	wire, err := json.Marshal(&genai.GenerateContentConfig{
		Tools: []*genai.Tool{{FunctionDeclarations: declarations}},
	})
	require.NoError(t, err)

	var payload struct {
		Tools []struct {
			FunctionDeclarations []struct {
				Parameters map[string]any `json:"parameters"`
			} `json:"functionDeclarations"`
		} `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(wire, &payload))
	parameters := payload.Tools[0].FunctionDeclarations[0].Parameters
	require.Equal(t, []any{"node", "choice"}, parameters["required"])

	properties := parameters["properties"].(map[string]any)
	node := properties["node"].(map[string]any)
	require.Equal(t, "OBJECT", node["type"])
	require.Equal(t, []any{"kind"}, node["required"])
	nodeProperties := node["properties"].(map[string]any)
	require.Equal(t, []any{"leaf", "branch"}, nodeProperties["kind"].(map[string]any)["enum"])
	require.NotContains(t, nodeProperties, "child")

	choice := properties["choice"].(map[string]any)
	anyOf := choice["anyOf"].([]any)
	require.Equal(t, []any{"a", "b"}, anyOf[0].(map[string]any)["enum"])
	require.Equal(t, []any{"count"}, anyOf[1].(map[string]any)["required"])
}

func TestToGoogleTools_InlinesReferences(t *testing.T) {
	t.Parallel()

	tools := []fantasy.Tool{
		fantasy.FunctionTool{
			Name: "refs",
			InputSchema: map[string]any{
				"type": "object",
				"definitions": map[string]any{
					"Filter": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"term": map[string]any{"type": "string"},
						},
						"required": []any{"term"},
					},
				},
				"$defs": map[string]any{
					"Text": map[string]any{"type": "string"},
				},
				"properties": map[string]any{
					"filter":  map[string]any{"$ref": "#/definitions/Filter", "description": "Search filter"},
					"encoded": map[string]any{"$ref": "#/%24defs/Text"},
					"list": map[string]any{
						"type":  "array",
						"items": map[string]any{"$ref": "#/$defs/Text"},
					},
					"self":     map[string]any{"$ref": "#"},
					"dangling": map[string]any{"$ref": "#/$defs/Missing"},
					"optional": map[string]any{
						"anyOf": []any{
							map[string]any{"type": "integer"},
							map[string]any{"type": "null"},
						},
					},
					"untyped": map[string]any{"description": "anything"},
				},
				"required": []any{"filter", "dangling", "self"},
			},
		},
	}

	declarations, _, warnings := toGoogleTools(tools, nil)
	require.Empty(t, warnings)
	require.Len(t, declarations, 1)
	parameters := declarations[0].Parameters

	require.Equal(t, genai.TypeObject, parameters.Type)
	// Recursive and unresolvable references are removed, together with
	// their required entries, rather than sent as strings.
	require.Equal(t, []string{"filter"}, parameters.Required)
	require.NotContains(t, parameters.Properties, "self")
	require.NotContains(t, parameters.Properties, "dangling")

	filter := parameters.Properties["filter"]
	require.Equal(t, genai.TypeObject, filter.Type)
	require.Equal(t, "Search filter", filter.Description)
	require.Equal(t, []string{"term"}, filter.Required)
	require.Equal(t, genai.TypeString, filter.Properties["term"].Type)

	require.Equal(t, genai.TypeString, parameters.Properties["encoded"].Type)
	require.Equal(t, genai.TypeString, parameters.Properties["list"].Items.Type)

	optional := parameters.Properties["optional"]
	require.Equal(t, genai.TypeInteger, optional.Type)
	require.NotNil(t, optional.Nullable)
	require.True(t, *optional.Nullable)
	require.Empty(t, optional.AnyOf)

	require.Equal(t, genai.TypeString, parameters.Properties["untyped"].Type)
}

func TestToGoogleTools_DropsArrayWithRecursiveItems(t *testing.T) {
	t.Parallel()

	tools := []fantasy.Tool{
		fantasy.FunctionTool{
			Name: "tree",
			InputSchema: map[string]any{
				"type": "object",
				"$defs": map[string]any{
					"Node": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name":     map[string]any{"type": "string"},
							"children": map[string]any{"type": "array", "items": map[string]any{"$ref": "#/$defs/Node"}},
						},
					},
				},
				"properties": map[string]any{
					"root": map[string]any{"$ref": "#/$defs/Node"},
				},
			},
		},
	}

	declarations, _, _ := toGoogleTools(tools, nil)
	root := declarations[0].Parameters.Properties["root"]
	require.Equal(t, genai.TypeObject, root.Type)
	require.Contains(t, root.Properties, "name")
	// An array whose items recurse cannot be expressed and would be
	// rejected without items, so the property is left out.
	require.NotContains(t, root.Properties, "children")
}
