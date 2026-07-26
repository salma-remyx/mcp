import { MondayAgentToolkit } from './toolkit';
import { Tool, ToolType } from '../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { ApiClient } from '@mondaydotcomorg/api';
import { z } from 'zod';
import { API_VERSION } from 'src/utils';

// Mock the API client (no network).
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));

// Control the registered tool set directly.
jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

function makeTool(name: string, description: string, schema: Record<string, any> = {}): Tool<any, any> {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] } as ToolAnnotations,
    enabledByDefault: true,
    getDescription: () => description,
    getInputSchema: () => schema as any,
    execute: async () => ({ content: '' }),
  };
}

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;
void ApiClient; // referenced for the mock factory above

const searchItemsTool = makeTool('search_items', 'Search for items and boards across the workspace', {
  query: z.string().describe('Free-text search query'),
  board_id: z.number().describe('Board identifier').optional(),
});
const createDocTool = makeTool('create_doc', 'Create a new document in a workspace', {
  title: z.string().describe('Document title'),
  content: z.string().describe('Document body content'),
});
const listBoardsTool = makeTool('list_boards', 'List all boards in the account');

describe('MondayAgentToolkit.selectToolsForIntent (Tool Attention gating)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetFilteredToolInstances.mockReturnValue([searchItemsTool, createDocTool, listBoardsTool]);
  });

  it('gates ON intent-relevant tools and gates OFF the rest via the dynamic manager', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    const selected = toolkit.selectToolsForIntent('search for items about the marketing board', { topK: 5 });
    const selectedNames = selected.map((s) => s.name);

    // The relevant tool is gated on...
    expect(selectedNames).toContain('search_items');
    // ...while the irrelevant document tool is gated off in the dynamic manager.
    const status = toolkit.getToolsStatus();
    expect(status['search_items']).toBe(true);
    expect(status['create_doc']).toBe(false);
  });

  it('keeps the management tool enabled regardless of intent overlap when the tool manager is on', () => {
    mockGetFilteredToolInstances.mockReturnValue([searchItemsTool, createDocTool, listBoardsTool]);
    const toolkit = new MondayAgentToolkit({
      mondayApiToken: 'test-token',
      toolsConfiguration: { enableToolManager: true },
    });

    // Intent that overlaps with no tool: only the always-included management tool survives.
    const selected = toolkit.selectToolsForIntent('hello world nothing matches here', { topK: 5 });
    const selectedNames = selected.map((s) => s.name);

    expect(selectedNames).toContain('manage_tools');
    const status = toolkit.getToolsStatus();
    expect(status['manage_tools']).toBe(true);
    expect(status['search_items']).toBe(false);
    expect(status['create_doc']).toBe(false);
  });

  it('does not depend on a transport (listChanged is skipped while disconnected)', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });
    // No transport is connected; gating must mutate state without throwing.
    expect(() => toolkit.selectToolsForIntent('search items', { topK: 1 })).not.toThrow();
    expect(toolkit.getServer().isConnected()).toBe(false);
    expect(API_VERSION).toBeDefined();
  });
});
