import { MondayAgentToolkit } from './toolkit';
import { ToolType } from '../core/tool';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';

// Mock the monday.com ApiClient so tool construction never touches the network.
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));

// Control which tool instances the toolkit sees.
jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

// Keep the management tool light when enableToolManager is on.
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

// Integration coverage for query-based tool gating wired into MondayAgentToolkit.getTools /
// getToolsForMcp. The non-new call-site module under test is ./toolkit (src/mcp/toolkit.ts).
describe('MondayAgentToolkit query-based tool gating', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const createDoc = {
    name: 'create_doc',
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: jest.fn().mockReturnValue('Create a new document in a workspace'),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn().mockResolvedValue({ content: 'ok' }),
  };

  const listBoards = {
    name: 'list_boards',
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: jest.fn().mockReturnValue('List all boards in the account'),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn().mockResolvedValue({ content: 'ok' }),
  };

  it('returns all tools when no query is given (eager behavior preserved)', () => {
    mockGetFilteredToolInstances.mockReturnValue([createDoc, listBoards]);
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    expect(toolkit.getTools()).toHaveLength(2);
  });

  it('gates getTools to the query-relevant tool, reducing the injected payload', () => {
    mockGetFilteredToolInstances.mockReturnValue([createDoc, listBoards]);
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    const gated = toolkit.getTools({ query: 'create a new document' });
    expect(gated).toHaveLength(1);
    expect(gated[0].name).toBe('create_doc');
  });

  it('gates getToolsForMcp the same way', () => {
    mockGetFilteredToolInstances.mockReturnValue([createDoc, listBoards]);
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    const gated = toolkit.getToolsForMcp({ query: 'list all boards' });
    expect(gated).toHaveLength(1);
    expect(gated[0].name).toBe('list_boards');
  });

  it('always keeps the management tool reachable even when it does not match the query', () => {
    mockGetFilteredToolInstances.mockReturnValue([createDoc, listBoards]);
    const toolkit = new MondayAgentToolkit({
      mondayApiToken: 'test-token',
      toolsConfiguration: { enableToolManager: true },
    });

    const gated = toolkit.getTools({ query: 'create a new document' });
    const names = gated.map((t) => t.name);
    expect(names).toContain('create_doc');
    expect(names).toContain('manage-tools');
    expect(names).not.toContain('list_boards');
  });

  it('falls back to all tools when the query matches nothing', () => {
    mockGetFilteredToolInstances.mockReturnValue([createDoc, listBoards]);
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });

    expect(toolkit.getTools({ query: 'weather forecast' })).toHaveLength(2);
  });
});
