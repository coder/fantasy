// Package filepart lets providers convert user file parts and tool-result
// media with the same rules.
package filepart

import (
	"encoding/base64"

	"charm.land/fantasy"
)

// File is a user file part or tool-result media. It keeps the content in the
// encoding its source uses, raw bytes or base64, so a converter encodes or
// decodes it only when the wire format needs the other one.
type File struct {
	MediaType       string
	Filename        string
	ProviderOptions fantasy.ProviderOptions

	raw     []byte
	encoded string
}

// FromFilePart returns the file for a file part.
func FromFilePart(part fantasy.FilePart) File {
	return File{
		MediaType:       part.MediaType,
		Filename:        part.Filename,
		ProviderOptions: part.ProviderOptions,
		raw:             part.Data,
	}
}

// FromToolResult returns the file for tool-result media.
func FromToolResult(media fantasy.ToolResultOutputContentMedia) File {
	return File{
		MediaType: media.MediaType,
		Filename:  media.Filename,
		encoded:   media.Data,
	}
}

// Base64 returns the content as standard base64.
func (f File) Base64() string {
	if f.raw == nil {
		return f.encoded
	}
	return base64.StdEncoding.EncodeToString(f.raw)
}

// Bytes returns the raw content. It fails only for tool-result media whose
// data is not valid base64.
func (f File) Bytes() ([]byte, error) {
	if f.raw != nil {
		return f.raw, nil
	}
	return base64.StdEncoding.DecodeString(f.encoded)
}
