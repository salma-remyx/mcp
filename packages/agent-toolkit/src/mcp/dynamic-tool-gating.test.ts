import { DynamicToolManager } from './dynamic-tool-manager';
import { Tool, ToolType } from '../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';
import { applyToolAttention, scoreToolRelevance, selectTools } from './dynamic-tool-gating';

// Mock MCP registration handle — mirrors the enable/disable surface the real
// MondayAgentToolkit hands to DynamicToolManager.registerTool().
interface MockMCPToolHandle {
  enable: jest.Mock;
  disable: jest.Mock;
}

interface MockTool extends Tool<any, any> {
  name: string;
  type: ToolType;
  annotations: ToolAnnotations;
  enabledByDefault?: boolean;
  getDescription: jest.Mock;
  getInputSchema: jest.Mock;
  execute: jest.Mock;
}

const makeTool = (name: string, description: string): MockTool => ({
  name,
  type: ToolType.READ,
  annotations: { audience: [] },
  enabledByDefault: true,
  getDescription: jest.fn().mockReturnValue(description),
  getInputSchema: jest.fn().mockReturnValue({}),
  execute: jest.fn(),
});

const makeHandle = (): MockMCPToolHandle => ({ enable: jest.fn(), disable: jest.fn() });

// A small, realistic slice of the toolkit's toolset — enough to exercise intent gating.
const FIXTURES: Array<{ name: string; description: string }> = [
  { name: 'create_item', description: 'Create a new item on a monday.com board' },
  { name: 'create_board', description: 'Create a new board in the workspace' },
  { name: 'search_items', description: 'Search for items and boards across the workspace' },
  { name: 'read_docs', description: 'Read and retrieve content from a monday.com document' },
  { name: 'update_doc', description: 'Update and edit blocks within an existing monday.com document' },
];

const registerAll = (manager: DynamicToolManager): Record<string, MockMCPToolHandle> => {
  const handles: Record<string, MockMCPToolHandle> = {};
  for (const fixture of FIXTURES) {
    handles[fixture.name] = makeHandle();
    manager.registerTool(makeTool(fixture.name, fixture.description), handles[fixture.name]);
  }
  return handles;
};

describe('scoreToolRelevance (parameter-free intent overlap)', () => {
  it('scores a tool whose name matches the intent higher than an unrelated one', () => {
    const intent = 'create a new item on the backlog board';
    const score = scoreToolRelevance(intent, { name: 'create_item', description: FIXTURES[0].description });
    const unrelated = scoreToolRelevance(intent, { name: 'read_docs', description: FIXTURES[3].description });
    expect(score).toBeGreaterThan(0);
    expect(score).toBeGreaterThan(unrelated);
    expect(unrelated).toBe(0);
  });

  it('returns 0 when the intent carries no signal tokens', () => {
    const score = scoreToolRelevance('the and of to', { name: 'create_item', description: 'create item' });
    expect(score).toBe(0);
  });
});

describe('selectTools (lazy-schema subset)', () => {
  const tools = FIXTURES.map((f) => ({ name: f.name, description: f.description }));

  it('keeps only the top-K relevant tools and gates the rest off', () => {
    const { selected, gatedOff } = selectTools('create a new item on the board', tools, { maxTools: 1 });
    expect(selected).toHaveLength(1);
    expect(selected[0].name).toBe('create_item');
    expect(gatedOff.map((t) => t.name)).toEqual(expect.arrayContaining(['read_docs', 'update_doc', 'search_items']));
  });

  it('force-includes alwaysOn tools without consuming the budget', () => {
    const { selected } = selectTools('create a new item on the board', tools, {
      maxTools: 1,
      alwaysOn: ['search_items'],
    });
    const names = selected.map((t) => t.name);
    expect(names).toEqual(expect.arrayContaining(['create_item', 'search_items']));
  });
});

describe('applyToolAttention (integration with DynamicToolManager)', () => {
  let manager: DynamicToolManager;
  let handles: Record<string, MockMCPToolHandle>;

  beforeEach(() => {
    manager = new DynamicToolManager();
    handles = registerAll(manager);
  });

  it('drives the existing manager: enables the relevant tool, disables the rest', () => {
    // All tools start enabled (simulating the eager full-schema tools/list baseline).
    expect(Object.values(manager.getToolsStatus()).every(Boolean)).toBe(true);

    const result = applyToolAttention(manager, 'create a new item on the board', { maxTools: 1 });

    // The intent-relevant tool stays on; the doc/search tools are gated off.
    expect(manager.isToolEnabled('create_item')).toBe(true);
    expect(manager.isToolEnabled('read_docs')).toBe(false);
    expect(manager.isToolEnabled('update_doc')).toBe(false);
    expect(manager.isToolEnabled('search_items')).toBe(false);

    // Disabled tools had their MCP handle.disable() invoked (schema drops out of tools/list).
    expect(handles.read_docs.disable).toHaveBeenCalled();
    expect(handles.update_doc.disable).toHaveBeenCalled();
    // The kept tool was never disabled.
    expect(handles.create_item.disable).not.toHaveBeenCalled();

    // The result reports exactly the state changes, not the full toolset.
    expect(result.enabled).not.toContain('create_item'); // already on, no change
    expect(result.disabled).toEqual(
      expect.arrayContaining(['read_docs', 'update_doc', 'search_items', 'create_board']),
    );
    // Every registered tool received a score for observability.
    expect(result.scores).toHaveLength(FIXTURES.length);
  });

  it('re-enables a previously gated-off tool when a later turn needs it', () => {
    applyToolAttention(manager, 'create a new item on the board', { maxTools: 1 });
    expect(manager.isToolEnabled('read_docs')).toBe(false);

    const result = applyToolAttention(manager, 'read the onboarding document', { maxTools: 1 });
    expect(manager.isToolEnabled('read_docs')).toBe(true);
    expect(manager.isToolEnabled('create_item')).toBe(false);
    expect(result.enabled).toContain('read_docs');
    expect(handles.read_docs.enable).toHaveBeenCalled();
  });

  it('preserves alwaysOn tools across turns regardless of intent', () => {
    const result = applyToolAttention(manager, 'create a new item on the board', {
      maxTools: 1,
      alwaysOn: ['search_items'],
    });
    expect(manager.isToolEnabled('search_items')).toBe(true);
    expect(result.scores.find((s) => s.name === 'search_items')?.score).toBe(1);
  });
});
