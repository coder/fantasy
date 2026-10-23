package fantasy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolResultOutputContentMediaFilePart(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		media   ToolResultOutputContentMedia
		want    FilePart
		wantErr string
	}{
		{
			name: "decodes data and copies metadata",
			media: ToolResultOutputContentMedia{
				Data:      "JVBERi0xLjQ=",
				MediaType: "application/pdf",
				Filename:  "report.pdf",
				Text:      "attached report",
			},
			want: FilePart{
				Filename:  "report.pdf",
				Data:      []byte("%PDF-1.4"),
				MediaType: "application/pdf",
			},
		},
		{
			name: "rejects invalid base64",
			media: ToolResultOutputContentMedia{
				Data:      "not base64!",
				MediaType: "application/pdf",
			},
			wantErr: "decode tool result media data",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := tt.media.FilePart()
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}
