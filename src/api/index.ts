/**
 * API module for LLM integration
 */

// Re-export the interfaces
export type {
	ModelApi,
	ModelResponse,
	BaseModelRequest,
	ExtendedModelRequest,
	ToolCall,
	ToolDefinition,
} from './interfaces/model-api';

// Export the simplified factory
export { GeminiClientFactory, ModelUseCase } from './simple-factory';

// Export the clients
export { GeminiClient } from './gemini-client';
export type { GeminiClientConfig } from './gemini-client';
export { OpenRouterClient } from './openrouter-client';
export type { OpenRouterClientConfig } from './openrouter-client';
