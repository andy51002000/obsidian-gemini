/**
 * OpenRouter API implementation using OpenAI-compatible chat completions
 */

import { requestUrl } from 'obsidian';
import {
	ModelApi,
	BaseModelRequest,
	ExtendedModelRequest,
	ModelResponse,
	ToolCall,
	StreamCallback,
	StreamingModelResponse,
	ToolDefinition,
} from './interfaces/model-api';
import { GeminiPrompts } from '../prompts';
import type ObsidianGemini from '../main';

const DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1';

type OpenRouterRole = 'system' | 'user' | 'assistant' | 'tool';

type OpenRouterContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } };

type OpenRouterMessage =
	| {
			role: 'system' | 'user' | 'assistant';
			content: string | OpenRouterContentPart[];
			tool_calls?: OpenRouterToolCall[];
	  }
	| {
			role: 'tool';
			tool_call_id: string;
			content: string;
	  };

type OpenRouterToolCall = {
	id: string;
	type: 'function';
	function: {
		name: string;
		arguments: string;
	};
};

/**
 * Configuration for OpenRouterClient
 */
export interface OpenRouterClientConfig {
	apiKey: string;
	model?: string;
	temperature?: number;
	topP?: number;
	maxOutputTokens?: number;
	streamingEnabled?: boolean;
	baseUrl?: string;
}

/**
 * OpenRouterClient - OpenAI-compatible chat completions wrapper
 *
 * Implements ModelApi while routing requests through OpenRouter.
 */
export class OpenRouterClient implements ModelApi {
	private config: OpenRouterClientConfig;
	private prompts: GeminiPrompts;
	private plugin?: ObsidianGemini;

	constructor(config: OpenRouterClientConfig, prompts?: GeminiPrompts, plugin?: ObsidianGemini) {
		this.config = {
			temperature: 1.0,
			topP: 0.95,
			streamingEnabled: true,
			baseUrl: DEFAULT_BASE_URL,
			...config,
		};
		this.plugin = plugin;
		this.prompts = prompts || new GeminiPrompts(plugin);
	}

	/**
	 * Generate a non-streaming response
	 */
	async generateModelResponse(request: BaseModelRequest | ExtendedModelRequest): Promise<ModelResponse> {
		const payload = await this.buildRequestPayload(request);
		const response = await this.requestOpenRouter(payload);
		return this.extractModelResponse(response);
	}

	/**
	 * Generate a streaming response
	 *
	 * OpenRouter streaming requires SSE support; we fall back to a single-shot response.
	 */
	generateStreamingResponse(
		request: BaseModelRequest | ExtendedModelRequest,
		onChunk: StreamCallback
	): StreamingModelResponse {
		let cancelled = false;

		const complete = (async (): Promise<ModelResponse> => {
			const response = await this.generateModelResponse(request);
			if (!cancelled && response.markdown) {
				onChunk({ text: response.markdown });
			}
			return response;
		})().catch((error) => {
			if (cancelled) {
				return { markdown: '', rendered: '' };
			}
			throw error;
		});

		return {
			complete,
			cancel: () => {
				cancelled = true;
			},
		};
	}

	/**
	 * Build the OpenRouter request payload
	 */
	private async buildRequestPayload(request: BaseModelRequest | ExtendedModelRequest): Promise<any> {
		const isExtended = 'userMessage' in request;
		const model = request.model || this.config.model;

		if (!model) {
			throw new Error('OpenRouter model not configured');
		}

		const systemPrompt = await this.buildSystemPrompt(request);
		const messages: OpenRouterMessage[] = [];

		if (systemPrompt) {
			messages.push({ role: 'system', content: systemPrompt });
		}

		if (isExtended) {
			const extReq = request as ExtendedModelRequest;
			const historyMessages = this.convertConversationHistory(extReq.conversationHistory || []);
			messages.push(...historyMessages);

			const userMessage = this.buildUserMessage(extReq.userMessage, extReq.imageAttachments);
			if (userMessage) {
				messages.push(userMessage);
			}
		} else if (request.prompt) {
			// Mirror Gemini behavior: prompt is both system instruction and user content
			messages.push({ role: 'user', content: request.prompt });
		}

		const payload: any = {
			model,
			messages,
			temperature: request.temperature ?? this.config.temperature,
			top_p: request.topP ?? this.config.topP,
		};

		if (this.config.maxOutputTokens) {
			payload.max_tokens = this.config.maxOutputTokens;
		}

		// Add tool definitions if available
		const tools = isExtended ? (request as ExtendedModelRequest).availableTools : undefined;
		if (tools && tools.length > 0) {
			payload.tools = this.convertTools(tools);
		}

		return payload;
	}

	/**
	 * Build system prompt (includes tools/custom prompt/agents memory)
	 */
	private async buildSystemPrompt(request: BaseModelRequest | ExtendedModelRequest): Promise<string> {
		const isExtended = 'userMessage' in request;

		if (!isExtended) {
			return request.prompt || '';
		}

		const extReq = request as ExtendedModelRequest;
		let agentsMemory: string | null = null;
		if (this.plugin?.agentsMemory) {
			try {
				agentsMemory = await this.plugin.agentsMemory.read();
			} catch (error) {
				this.plugin.logger.warn('Failed to load AGENTS.md:', error);
			}
		}

		let systemInstruction = this.prompts.getSystemPromptWithCustom(
			extReq.availableTools,
			extReq.customPrompt,
			agentsMemory
		);

		if (extReq.prompt && !extReq.customPrompt?.overrideSystemPrompt) {
			systemInstruction += '\n\n' + extReq.prompt;
		}

		return systemInstruction;
	}

	/**
	 * Build user message content (text + images)
	 */
	private buildUserMessage(
		userMessage: string,
		images?: { base64: string; mimeType: string }[]
	): OpenRouterMessage | null {
		const hasText = userMessage && userMessage.trim();
		const hasImages = images && images.length > 0;

		if (!hasText && !hasImages) {
			return null;
		}

		if (!hasImages) {
			return { role: 'user', content: userMessage };
		}

		const contentParts: OpenRouterContentPart[] = [];
		if (hasText) {
			contentParts.push({ type: 'text', text: userMessage });
		}

		images!.forEach((img) => {
			contentParts.push({
				type: 'image_url',
				image_url: {
					url: `data:${img.mimeType};base64,${img.base64}`,
				},
			});
		});

		return {
			role: 'user',
			content: contentParts,
		};
	}

	/**
	 * Convert conversation history into OpenRouter messages
	 */
	private convertConversationHistory(history: any[]): OpenRouterMessage[] {
		const messages: OpenRouterMessage[] = [];
		let toolCallIndex = 0;
		let pendingToolCalls: Array<{ id: string; name: string }> = [];

		const createToolCallId = () => `call_${toolCallIndex++}`;

		for (const entry of history) {
			if (!entry) continue;

			// Gemini Content format (role + parts)
			if (entry.role && Array.isArray(entry.parts)) {
				const role: OpenRouterRole = entry.role === 'user' ? 'user' : 'assistant';
				const parts = entry.parts as any[];
				const textParts = parts.filter((part) => typeof part.text === 'string' && part.text.length > 0);
				const functionCalls = parts.filter((part) => part.functionCall);
				const functionResponses = parts.filter((part) => part.functionResponse);

				if (functionCalls.length > 0) {
					const toolCalls: OpenRouterToolCall[] = functionCalls.map((part) => {
						const id = createToolCallId();
						const name = part.functionCall?.name || 'unknown';
						pendingToolCalls.push({ id, name });
						return {
							id,
							type: 'function',
							function: {
								name,
								arguments: JSON.stringify(part.functionCall?.args || {}),
							},
						};
					});

					const content = textParts.map((part) => part.text).join('');
					messages.push({
						role: 'assistant',
						content: content || '',
						tool_calls: toolCalls,
					});
					continue;
				}

				if (functionResponses.length > 0) {
					for (const part of functionResponses) {
						const responseName = part.functionResponse?.name || 'unknown';
						const pendingIndex = pendingToolCalls.findIndex((pending) => pending.name === responseName);
						const pending = pendingIndex >= 0 ? pendingToolCalls.splice(pendingIndex, 1)[0] : pendingToolCalls.shift();
						const toolCallId = pending?.id || createToolCallId();
						const content = this.safeStringify(part.functionResponse?.response ?? {});
						messages.push({
							role: 'tool',
							tool_call_id: toolCallId,
							content,
						});
					}

					if (textParts.length > 0) {
						messages.push({
							role,
							content: textParts.map((part) => part.text).join(''),
						});
					}
					continue;
				}

				if (textParts.length > 0) {
					messages.push({
						role,
						content: textParts.map((part) => part.text).join(''),
					});
				}
				continue;
			}

			// Simple role + text/message format
			if (entry.role && (entry.text || entry.message)) {
				const role: OpenRouterRole = entry.role === 'user' ? 'user' : 'assistant';
				const content = entry.text || entry.message || '';
				if (content) {
					messages.push({ role: role as 'user' | 'assistant', content });
				}
			}
		}

		return messages;
	}

	/**
	 * Convert tool definitions to OpenAI-compatible format
	 */
	private convertTools(tools: ToolDefinition[]): Array<{ type: 'function'; function: any }> {
		return tools.map((tool) => ({
			type: 'function' as const,
			function: {
				name: tool.name,
				description: tool.description,
				parameters: tool.parameters,
			},
		}));
	}

	/**
	 * Execute OpenRouter request
	 */
	private async requestOpenRouter(payload: any): Promise<any> {
		const baseUrl = this.config.baseUrl || DEFAULT_BASE_URL;
		const url = `${baseUrl.replace(/\/$/, '')}/chat/completions`;

		const response = await requestUrl({
			url,
			method: 'POST',
			headers: {
				Authorization: `Bearer ${this.config.apiKey}`,
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(payload),
			throw: false,
		});

		if (response.status >= 400) {
			const errorMessage =
				(response.json && (response.json as any)?.error?.message) || response.text || `HTTP ${response.status}`;
			const error = new Error(errorMessage);
			(error as any).status = response.status;
			throw error;
		}

		return response.json || JSON.parse(response.text || '{}');
	}

	/**
	 * Extract ModelResponse from OpenRouter response
	 */
	private extractModelResponse(response: any): ModelResponse {
		const message = response?.choices?.[0]?.message;
		const content = message?.content;

		let markdown = '';
		if (Array.isArray(content)) {
			markdown = content
				.filter((part) => part.type === 'text')
				.map((part) => part.text)
				.join('');
		} else if (typeof content === 'string') {
			markdown = content;
		}

		const toolCalls = this.extractToolCalls(message);

		return {
			markdown,
			rendered: '',
			...(toolCalls && { toolCalls }),
		};
	}

	/**
	 * Extract tool calls from response message
	 */
	private extractToolCalls(message: any): ToolCall[] | undefined {
		const toolCalls = message?.tool_calls;
		if (!Array.isArray(toolCalls) || toolCalls.length === 0) {
			// Legacy function_call field
			if (message?.function_call?.name) {
				return [
					{
						name: message.function_call.name,
						arguments: this.parseToolArguments(message.function_call.arguments),
					},
				];
			}
			return undefined;
		}

		const calls: ToolCall[] = [];
		for (const call of toolCalls) {
			const name = call?.function?.name;
			if (!name) continue;
			calls.push({
				name,
				arguments: this.parseToolArguments(call.function.arguments),
			});
		}

		return calls.length > 0 ? calls : undefined;
	}

	private parseToolArguments(args: any): Record<string, any> {
		if (!args) return {};
		if (typeof args === 'object') {
			return args;
		}
		if (typeof args === 'string') {
			try {
				return JSON.parse(args);
			} catch (error) {
				this.plugin?.logger.warn('[OpenRouterClient] Failed to parse tool arguments JSON, using raw string');
				return { _raw: args };
			}
		}
		return {};
	}

	private safeStringify(value: any): string {
		if (typeof value === 'string') {
			return value;
		}
		try {
			return JSON.stringify(value);
		} catch (error) {
			this.plugin?.logger.warn('[OpenRouterClient] Failed to stringify tool response');
			return String(value);
		}
	}
}
