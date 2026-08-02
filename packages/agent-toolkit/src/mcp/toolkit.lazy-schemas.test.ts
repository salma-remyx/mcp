import { MondayAgentToolkit } from './toolkit';
import { ToolType } from '../core/tool';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { z } from 'zod';

// Mock the ApiClient (constructed during toolkit init)
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));

// Control the registered tool surface so the gate is deterministic
jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;

// Three tools with prose-heavy JSON schemas — the bulk that lazy loading trims.
const buildTools = () => [
  {
    name: 'create_column',
    type: ToolType.WRITE,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: () => 'Create a new column in a monday.com board',
    getInputSchema: () => ({
      columnTitle: z.string().describe('The title of the column to be created'),
      columnType: z.string().describe('The type of the column to be created'),
    }),
    execute: jest.fn(),
  },
  {
    name: 'list_boards',
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: () => 'List boards in a monday.com workspace',
    getInputSchema: () => ({
      limit: z.number().describe('Maximum number of boards to return'),
    }),
    execute: jest.fn(),
  },
  {
    name: 'search_items',
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: () => 'Search for items across the workspace',
    getInputSchema: () => ({
      query: z.string().describe('Search query text'),
    }),
    execute: jest.fn(),
  },
];

const buildToolkit = () => {
  mockGetFilteredToolInstances.mockReturnValue(buildTools() as any);
  return new MondayAgentToolkit({ mondayApiToken: 'test-token' });
};

describe('MondayAgentToolkit lazy schema loading', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('emits full JSON schemas for every tool by default (no lazySchemas)', () => {
    const toolkit = buildToolkit();
    const tools = toolkit.getToolsForMcp({ schemaFormat: 'json' });

    expect(tools).toHaveLength(3);
    for (const tool of tools) {
      expect(tool.schema.properties).toBeDefined();
      // Full schemas retain the per-field description prose
      const firstProp = Object.keys(tool.schema.properties)[0];
      expect(tool.schema.properties[firstProp].description).toBeDefined();
    }
  });

  it('promotes the full schema only for the relevant tool and summarizes the rest', () => {
    const toolkit = buildToolkit();
    const tools = toolkit.getToolsForMcp({
      schemaFormat: 'json',
      lazySchemas: { query: 'create a column', maxFullSchemas: 1 },
    });

    expect(tools).toHaveLength(3); // nothing dropped — summaries keep tools visible

    const byName = Object.fromEntries(tools.map((tool) => [tool.name, tool.schema]));

    // Relevant tool keeps full schema (prose preserved)
    expect(byName.create_column.properties.columnTitle.description).toBe('The title of the column to be created');

    // Non-relevant tools get summary schemas: structure kept, prose trimmed
    expect(byName.list_boards.properties.limit.type).toBe('number');
    expect(byName.list_boards.properties.limit.description).toBeUndefined();
    expect(byName.search_items.properties.query.type).toBe('string');
    expect(byName.search_items.properties.query.description).toBeUndefined();
  });

  it('does not alter the default Zod-shape path when lazySchemas is set', () => {
    const toolkit = buildToolkit();
    const tools = toolkit.getTools({
      lazySchemas: { query: 'create a column', maxFullSchemas: 1 },
    });

    // Without schemaFormat: 'json', the raw Zod shape is returned untouched
    expect(tools[0].schema.columnTitle).toBeDefined();
    expect(tools[0].schema.columnTitle.description).toBe('The title of the column to be created');
  });
});
