import { z } from 'zod';
import { Tool, ToolType } from '../core/tool';
import { DynamicToolManager } from './dynamic-tool-manager';
import {
  ToolGatingRouter,
  estimatePayloadTokens,
  scoreTool,
  selectTools,
  serializeToolSchema,
  tokenizeIntent,
} from './tool-gating-router';

// Minimal handle the DynamicToolManager drives (mirrors MCPToolHandle).
interface MockHandle {
  enable: jest.Mock;
  disable: jest.Mock;
}

function makeTool(name: string, type: ToolType, description: string, schema?: Record<string, unknown>): Tool<any, any> {
  return {
    name,
    type,
    annotations: { title: name },
    enabledByDefault: true,
    getDescription: () => description,
    getInputSchema: () => schema as any,
    execute: jest.fn(),
  } as unknown as Tool<any, any>;
}

// A small slice of the real monday.com tool catalog, with realistic names.
function buildCatalog(): Tool<any, any>[] {
  return [
    makeTool('search_items', ToolType.READ, 'Search for items and boards across the workspace by query.', {
      query: z.string().describe('The search query text'),
      limit: z.number().optional().describe('Maximum number of results to return'),
    }),
    makeTool('create_doc', ToolType.WRITE, 'Create a new document inside a workspace.', {
      workspace_id: z.number().describe('Target workspace id'),
      title: z.string().describe('Document title'),
      content: z.string().describe('Initial document content'),
    }),
    makeTool('update_doc', ToolType.WRITE, 'Update a specific block inside an existing document.', {
      doc_id: z.number().describe('Document to update'),
      block_id: z.string().describe('Block to edit'),
      content: z.string().describe('New block content'),
    }),
    makeTool('list_boards', ToolType.READ, 'List all boards visible in the account.', {
      limit: z.number().optional().describe('Maximum boards to return'),
    }),
    makeTool('get_user_context', ToolType.READ, 'Get the current user context, teams and active boards.'),
  ];
}

describe('tool-gating-router', () => {
  describe('scoreTool', () => {
    it('weights name-token matches above description matches', () => {
      const tools = buildCatalog();
      const createDoc = tools.find((t) => t.name === 'create_doc')!;
      // "create" and "doc" are both name tokens -> 3 each; "new" only in description -> 1.
      expect(scoreTool('create a new doc', createDoc)).toBe(7);
    });

    it('returns 0 for tools unrelated to the intent', () => {
      const tools = buildCatalog();
      const searchItems = tools.find((t) => t.name === 'search_items')!;
      expect(scoreTool('create a new doc', searchItems)).toBe(0);
    });

    it('returns 0 when the intent has no usable terms', () => {
      expect(scoreTool('the and for', buildCatalog()[0])).toBe(0);
    });
  });

  describe('tokenizeIntent', () => {
    it('drops stopwords and short tokens, lowercases, and dedupes', () => {
      expect(tokenizeIntent('Search for ITEMS, about items!')).toEqual(['search', 'items']);
    });
  });

  describe('selectTools', () => {
    it('returns only the intent-relevant subset, ranked', () => {
      const tools = buildCatalog();
      const picked = selectTools('search for items about Q3', tools);
      expect(picked.map((t) => t.name)).toEqual(['search_items']);
    });

    it('keeps the whole tool family when the intent is broad to it', () => {
      const tools = buildCatalog();
      const picked = selectTools('create a new doc', tools);
      const names = picked.map((t) => t.name);
      expect(names).toContain('create_doc');
      expect(names).toContain('update_doc');
      expect(names).not.toContain('search_items');
    });

    it('always includes safety-net tool names even with a low score', () => {
      const tools = buildCatalog();
      const picked = selectTools('search for items about Q3', tools, { alwaysInclude: ['manage_tools'] });
      // manage_tools is not in the catalog, so only the matched tool is present.
      expect(picked.map((t) => t.name)).toEqual(['search_items']);
    });
  });

  describe('serializeToolSchema / estimatePayloadTokens', () => {
    it('serializes a zod schema to JSON schema lazily', () => {
      const tool = buildCatalog().find((t) => t.name === 'create_doc')!;
      const schema = serializeToolSchema(tool) as { properties: unknown; type: string };
      expect(schema.type).toBe('object');
      expect(schema.properties).toHaveProperty('workspace_id');
      expect(schema.properties).toHaveProperty('title');
    });

    it('estimates a larger token payload for the full catalog than for a subset', () => {
      const tools = buildCatalog();
      const full = estimatePayloadTokens(tools);
      const subset = estimatePayloadTokens(tools.slice(0, 1));
      expect(full).toBeGreaterThan(subset);
      expect(full).toBeGreaterThan(0);
    });
  });

  describe('ToolGatingRouter.route (integration with DynamicToolManager)', () => {
    let manager: DynamicToolManager;
    let handles: Record<string, MockHandle>;

    beforeEach(() => {
      manager = new DynamicToolManager();
      handles = {};
      for (const tool of buildCatalog()) {
        const handle: MockHandle = { enable: jest.fn(), disable: jest.fn() };
        handles[tool.name] = handle;
        manager.registerTool(tool, handle);
      }
    });

    it('enables only the gated subset and disables the rest via the manager', () => {
      const router = new ToolGatingRouter(manager, { minScore: 1 });
      const result = router.route('create a new doc', buildCatalog());

      expect(result.selected).toContain('create_doc');
      // Non-doc tools were disabled through the existing DynamicToolManager.
      expect(result.disabled).toEqual(expect.arrayContaining(['search_items', 'list_boards', 'get_user_context']));
      expect(result.disabled).not.toContain('create_doc');
      // Selected tool stays enabled (no disable call); others were disabled exactly once.
      expect(handles['create_doc'].disable).not.toHaveBeenCalled();
      expect(handles['search_items'].disable).toHaveBeenCalledTimes(1);
    });

    it('materializes lazy schemas only for the gated subset', () => {
      const router = new ToolGatingRouter(manager);
      const result = router.route('search for items about Q3', buildCatalog());

      expect(Object.keys(result.schemas)).toEqual(['search_items']);
    });

    it('reports the MCP-Tax reduction (tokensAfter < tokensBefore)', () => {
      const router = new ToolGatingRouter(manager);
      const result = router.route('search for items about Q3', buildCatalog());

      expect(result.tokensBefore).toBeGreaterThan(result.tokensAfter);
      expect(result.tokensSaved).toBe(result.tokensBefore - result.tokensAfter);
      expect(result.tokensSaved).toBeGreaterThan(0);
      expect(result.fellBack).toBe(false);
    });

    it('falls back to the full catalog when no tool matches the intent', () => {
      const router = new ToolGatingRouter(manager);
      const result = router.route('compute the mandelbrot set', buildCatalog());

      expect(result.fellBack).toBe(true);
      expect(result.selected).toHaveLength(5);
      expect(result.disabled).toEqual([]);
      expect(result.tokensSaved).toBe(0);
    });
  });
});
