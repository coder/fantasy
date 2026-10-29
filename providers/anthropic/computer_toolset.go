package anthropic

import (
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"charm.land/fantasy"
	"github.com/charmbracelet/anthropic-sdk-go"
)

// computerToolsetName is the toolset_name that marks tool_use and
// tool_result blocks as members of the computer use toolset.
const computerToolsetName = "computer"

// computerToolsetReplayFallbackMember is the member a replayed
// computer call is sent as when its input has no usable action.
const computerToolsetReplayFallbackMember = string(ActionScreenshot)

// isComputerToolsetMember reports whether a tool_use block is a
// computer use toolset member call. The SDK has no typed
// toolset_name field, so it is read from the block's extra fields.
func isComputerToolsetMember(block anthropic.ToolUseBlock) bool {
	return block.JSON.ExtraFields["toolset_name"].Raw() == `"`+computerToolsetName+`"`
}

// computerToolsetCallInput returns a member call's input in the
// legacy computer tool shape: the member input object with an
// "action" field naming the member. Input that is not a JSON object,
// such as arguments truncated at max_tokens, is returned unchanged
// so the call is still reported as invalid instead of being guessed.
func computerToolsetCallInput(member, input string) string {
	fields := map[string]any{}
	if strings.TrimSpace(input) != "" {
		var raw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(input), &raw); err != nil {
			return input
		}
		for k, v := range raw {
			fields[k] = v
		}
	}
	fields["action"] = member
	merged, err := json.Marshal(fields)
	if err != nil {
		return input
	}
	return string(merged)
}

// declaresComputerToolset reports whether tools include the computer
// use tool at version ComputerUseToolset20260801.
func declaresComputerToolset(tools []fantasy.Tool) bool {
	for _, tool := range tools {
		pdt, ok := asProviderDefinedTool(tool)
		if !ok || pdt.ID != computerUseToolID {
			continue
		}
		if version, ok := getComputerUseVersion(pdt); ok && version == ComputerUseToolset20260801 {
			return true
		}
	}
	return false
}

// replayComputerToolsetCalls rewrites replayed computer calls from
// the legacy shape (name "computer", input with "action") into
// toolset member calls, and tags every tool_result answering one
// with toolset_name. The API rejects a member call or result that
// lacks toolset_name. It edits messages in place.
func replayComputerToolsetCalls(messages []anthropic.MessageParam) []fantasy.CallWarning {
	var warnings []fantasy.CallWarning
	memberCalls := map[string]struct{}{}
	for _, msg := range messages {
		for _, block := range msg.Content {
			switch {
			case block.OfToolUse != nil && block.OfToolUse.Name == computerUseAPIName:
				toolUse := block.OfToolUse
				member, input, ok := computerToolsetMemberCall(toolUse.Input)
				if !ok {
					// Every tool_result must answer a tool_use with the
					// same ID, so the call cannot be dropped. screenshot
					// takes no input, so it is a valid member call for
					// any history; the result keeps its pairing.
					member, input = computerToolsetReplayFallbackMember, map[string]any{}
					warnings = append(warnings, fantasy.CallWarning{
						Type: fantasy.CallWarningTypeOther,
						Message: fmt.Sprintf(
							"computer tool call %q has no string action; replaying it as the %s toolset member to preserve tool_use and tool_result pairing",
							toolUse.ID, computerToolsetReplayFallbackMember,
						),
					})
				}
				toolUse.Name = member
				toolUse.Input = input
				toolUse.SetExtraFields(map[string]any{"toolset_name": computerToolsetName})
				memberCalls[toolUse.ID] = struct{}{}
			case block.OfToolResult != nil:
				if _, ok := memberCalls[block.OfToolResult.ToolUseID]; ok {
					block.OfToolResult.SetExtraFields(map[string]any{"toolset_name": computerToolsetName})
				}
			}
		}
	}
	return warnings
}

// computerToolsetMemberCall splits a legacy computer call input into
// the member name (its action) and the member input (the rest).
func computerToolsetMemberCall(input any) (member string, memberInput map[string]any, ok bool) {
	fields, ok := input.(map[string]any)
	if !ok {
		return "", nil, false
	}
	action, ok := fields["action"].(string)
	if !ok || action == "" {
		return "", nil, false
	}
	memberInput = maps.Clone(fields)
	delete(memberInput, "action")
	return action, memberInput, true
}
