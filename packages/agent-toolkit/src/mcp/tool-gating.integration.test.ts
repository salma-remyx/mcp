import { MondayAgentToolkit } from './toolkit';
import { ToolType } from '../core/tool';
import { ApiClient } from '@mondaydotcomorg/api';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';

// Same mock shape the existing toolkit.test.ts uses: the ApiClient and the
// tool factory are stubbed so we control exactly which tools register.
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

jest.mock('../core/tools/platform-api-tools/manage-tools-tool', () => ({
  ManageToolsTool: jest.fn().mockImplementation(() => ({
    name: 'manage-tools',
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: jest.fn().mockReturnValue('Manage tools'),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn(),
    setToolkitManager: jest.fn(),
  })),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;
const mockApiClient = ApiClient as jest.MockedClass<typeof ApiClient>;

describe('MondayAgentToolkit.gateToolsForQuery (Tool Attention wiring)', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    const createItem = {
      name: 'create_item',
      type: ToolType.WRITE,
      annotations: { audience: [] },
      enabledByDefault: true,
      getDescription: jest.fn().mockReturnValue('Create a new item on a board'),
      getInputSchema: jest.fn().mockReturnValue({}),
      execute: jest.fn(),
    };
    const searchItems = {
      name: 'search_items',
      type: ToolType.READ,
      annotations: { audience: [] },
      enabledByDefault: true,
      getDescription: jest.fn().mockReturnValue('Search for items and boards across the workspace'),
      getInputSchema: jest.fn().mockReturnValue({}),
      execute: jest.fn(),
    };
    const readDocs = {
      name: 'read_docs',
      type: ToolType.READ,
      annotations: { audience: [] },
      enabledByDefault: true,
      getDescription: jest.fn().mockReturnValue('Read and retrieve content from documents'),
      getInputSchema: jest.fn().mockReturnValue({}),
      execute: jest.fn(),
    };

    mockGetFilteredToolInstances.mockReturnValue([createItem, searchItems, readDocs]);
  });

  it('builds the toolkit with the three mock tools registered', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    expect(mockApiClient).toHaveBeenCalled();
    expect(toolkit.getDynamicToolNames().sort()).toEqual(['create_item', 'read_docs', 'search_items']);
    // All start enabled by default.
    expect(toolkit.isToolEnabled('create_item')).toBe(true);
    expect(toolkit.isToolEnabled('search_items')).toBe(true);
    expect(toolkit.isToolEnabled('read_docs')).toBe(true);
  });

  it('keeps the relevant tool enabled and defers the rest via the existing enable/disable plumbing', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    const plan = toolkit.gateToolsForQuery('create a new item on the board', { maxTools: 1 });

    // The gating plan reflects the relevance ranking.
    expect(plan.enabled).toContain('create_item');
    expect(plan.disabled).toContain('search_items');
    expect(plan.disabled).toContain('read_docs');
    expect(plan.estimatedSavedTokens).toBeGreaterThan(0);

    // The wiring actually toggled the dynamic tool manager state.
    expect(toolkit.isToolEnabled('create_item')).toBe(true);
    expect(toolkit.isToolEnabled('search_items')).toBe(false);
    expect(toolkit.isToolEnabled('read_docs')).toBe(false);
  });

  it('re-enables a previously deferred tool on a later, more relevant query', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    // First turn: only the item tool is relevant.
    toolkit.gateToolsForQuery('create a new item on the board', { maxTools: 1 });
    expect(toolkit.isToolEnabled('search_items')).toBe(false);

    // Next turn: the agent asks to search — the gate flips it back on.
    toolkit.gateToolsForQuery('search for items', { maxTools: 1 });
    expect(toolkit.isToolEnabled('search_items')).toBe(true);
  });
});
