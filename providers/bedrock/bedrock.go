// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock's language models.
package bedrock

import (
	"context"
	"os"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/charmbracelet/anthropic-sdk-go/option"
)

type options struct {
	region           string
	skipAuth         bool
	hasAPIKey        bool
	anthropicOptions []anthropic.Option
}

type provider struct {
	inner  fantasy.Provider
	region string
	// useDefaultChain reports whether the inner Anthropic Bedrock provider will
	// resolve the request region via the AWS default credential chain (rather
	// than the static API-key/skip-auth config). It mirrors the inner provider's
	// auth-mode decision so model-ID region prefixing matches the request.
	useDefaultChain bool
}

const (
	// Name is the name of the Bedrock provider.
	Name = "bedrock"
)

// Option defines a function that configures Bedrock provider options.
type Option = func(*options)

// New creates a new Bedrock provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}

	region := strings.TrimSpace(o.region)
	anthropicOptions := append([]anthropic.Option(nil), o.anthropicOptions...)
	if region == "" {
		region = strings.TrimSpace(os.Getenv("AWS_REGION"))
		if region != "" {
			anthropicOptions = append(anthropicOptions, anthropic.WithBedrockRegion(region))
		}
	}

	inner, err := anthropic.New(
		append(
			anthropicOptions,
			anthropic.WithName(Name),
			anthropic.WithBedrock(),
			anthropic.WithSkipAuth(o.skipAuth),
		)...,
	)
	if err != nil {
		return nil, err
	}
	// Mirror the inner Anthropic Bedrock provider's auth-mode decision
	// (skipAuth || apiKey != "") so legacy model-ID region prefixing resolves the
	// region the same way the request is signed for.
	useDefaultChain := !o.skipAuth && !o.hasAPIKey
	return &provider{inner: inner, region: region, useDefaultChain: useDefaultChain}, nil
}

func (p *provider) Name() string {
	return Name
}

func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	return p.inner.LanguageModel(ctx, normalizeModelID(ctx, modelID, p.region, p.useDefaultChain))
}

// WithAPIKey sets the access token for the Bedrock provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		// Track the final auth mode (last write wins, matching the inner
		// anthropic provider) so model-ID region resolution can mirror it.
		o.hasAPIKey = apiKey != ""
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithAPIKey(apiKey))
	}
}

// WithHeaders sets the headers for the Bedrock provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the Bedrock provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHTTPClient(client))
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithUserAgent(ua))
	}
}

// WithBaseURL sets the base URL for the Bedrock provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithBaseURL(baseURL))
	}
}

// WithSkipAuth configures whether to skip authentication for the Bedrock provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}

// WithRegion sets the AWS region for the Bedrock provider.
func WithRegion(region string) Option {
	return func(o *options) {
		o.region = region
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithBedrockRegion(region))
	}
}
