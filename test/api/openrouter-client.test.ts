import { requestUrl } from 'obsidian';
import { OpenRouterClient, OpenRouterClientConfig } from '../../src/api/openrouter-client';
import { GeminiPrompts } from '../../src/prompts';

// Mock window.localStorage
const mockLocalStorage = {
	getItem: jest.fn().mockReturnValue('en'),
	setItem: jest.fn(),
	removeItem: jest.fn(),
	clear: jest.fn(),
};
Object.defineProperty(window, 'localStorage', {
	value: mockLocalStorage,
	writable: true,
});

describe('OpenRouterClient', () => {
	const mockRequestUrl = requestUrl as jest.Mock;
	let mockPlugin: any;
	let client: OpenRouterClient;

	beforeEach(() => {
		mockRequestUrl.mockReset();

		mockPlugin = {
			logger: {
				log: jest.fn(),
				debug: jest.fn(),
				error: jest.fn(),
				warn: jest.fn(),
			},
			settings: {
				userName: 'User',
				ragIndexing: { enabled: false },
			},
		};

		const config: OpenRouterClientConfig = {
			apiKey: 'test-key',
			model: 'openrouter/test-model',
		};

		const prompts = new GeminiPrompts(mockPlugin);
		client = new OpenRouterClient(config, prompts, mockPlugin);
	});

	test('builds tool call history and tool responses in request payload', async () => {
		mockRequestUrl.mockResolvedValue({
			status: 200,
			json: {
				choices: [
					{
						message: {
							content: 'ok',
						},
					},
				],
			},
		});

		await client.generateModelResponse({
			userMessage: 'Continue',
			conversationHistory: [
				{ role: 'user', parts: [{ text: 'Hi' }] },
				{ role: 'model', parts: [{ functionCall: { name: 'read_file', args: { path: 'note.md' } } }] },
				{ role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { success: true } } }] },
			],
			prompt: '',
			availableTools: [
				{
					name: 'read_file',
					description: 'Read a file',
					parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
				},
			],
		});

		const payload = JSON.parse(mockRequestUrl.mock.calls[0][0].body as string);
		expect(payload.model).toBe('openrouter/test-model');
		expect(payload.tools).toHaveLength(1);

		const assistantWithTools = payload.messages.find((msg: any) => msg.role === 'assistant' && msg.tool_calls);
		expect(assistantWithTools).toBeTruthy();
		expect(assistantWithTools.tool_calls[0].function.name).toBe('read_file');

		const toolResponse = payload.messages.find((msg: any) => msg.role === 'tool');
		expect(toolResponse).toBeTruthy();
	});

	test('parses tool calls from OpenRouter response', async () => {
		mockRequestUrl.mockResolvedValue({
			status: 200,
			json: {
				choices: [
					{
						message: {
							content: '',
							tool_calls: [
								{
									id: 'call_1',
									type: 'function',
									function: {
										name: 'list_files',
										arguments: '{"path":"."}',
									},
								},
							],
						},
					},
				],
			},
		});

		const response = await client.generateModelResponse({
			userMessage: 'List files',
			conversationHistory: [],
			prompt: '',
		});

		expect(response.toolCalls).toEqual([{ name: 'list_files', arguments: { path: '.' } }]);
	});
});
